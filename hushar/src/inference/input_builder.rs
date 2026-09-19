// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Turning a request's features into the model's inputs.
//!
//! Everything that does not depend on a request is settled once, at startup, by
//! [`InputBuilder::resolve`]: which feature feeds which input, what each input's element
//! type is, and how many values per row it carries. A request then only walks that list.
//!
//! The configuration declares the mapping; it is checked against the model's signature,
//! never inferred from it.
//!
//! > With `feature_order`, features are transformed and concatenated into one input.
//! > With `model_inputs`, every input is copied from the request verbatim — and a feature
//! > a row omits is refused rather than defaulted.
//!
//! The mode decides, not the input's type: the transformations live inside
//! [`Source::Vectorised`], so a verbatim input cannot reach one. Transformations produce
//! floats, so only a concatenated float input can be built from them.

use std::collections::HashMap;
use std::ops::Range;

use crate::config::vectorization_config::{InputDataType, InputMode, VectorizationConfig};
use crate::inference::InferenceError;
use crate::inference::batch::{ElementType, FeatureData, InputBatch, IoSpec};
use crate::inference::proto_columns;
use crate::inference::transformations::Transformation;
use hushar::hushar_proto::Feature;

/// Where one model input's values come from.
///
/// # Fields
///
/// * `Vectorised` - every feature named, paired with the transformation that builds it,
///   concatenated in that order
/// * `Feature` - the one feature of that name, copied from the request as sent
#[derive(Debug)]
enum Source {
    Vectorised(Vec<(String, Box<dyn Transformation>)>),
    Feature(String),
}

/// One model input, fully settled.
///
/// # Fields
///
/// * `name` - the model's name for this input
/// * `element` - its element type, which chooses the buffer to fill. It does **not**
///   decide whether the input is transformed; `source` does
/// * `width` - values per row, settled at startup from the model or the configuration
/// * `source` - where the values come from
#[derive(Debug)]
struct PlannedInput {
    name: String,
    element: ElementType,
    width: usize,
    source: Source,
}

/// Builds a batch of model inputs from a batch of feature rows.
///
/// Each input carries its own source, so building needs nothing but a request.
#[derive(Debug)]
pub(crate) struct InputBuilder {
    inputs: Vec<PlannedInput>,
}

impl InputBuilder {
    /// Checks a configuration against a model's signature and settles the mapping.
    ///
    /// `feature_order` concatenates every feature into the model's one input;
    /// `model_inputs` feeds one input per named feature, verbatim.
    ///
    /// # Errors
    ///
    /// Fails if the model declares no inputs, or if the configuration disagrees with the
    /// signature over a name, an element type or a width.
    pub(crate) fn resolve(
        config: VectorizationConfig,
        specs: &[IoSpec],
    ) -> Result<Self, InferenceError> {
        if specs.is_empty() {
            return Err("the model declares no inputs, so there is nothing to feed".into());
        }

        match config.mode {
            InputMode::Vectorised {
                feature_order,
                data_type,
                transformations,
            } => Self::vectorised(transformations, feature_order, data_type, specs),
            InputMode::Named { model_inputs } => Self::named(model_inputs, specs),
        }
    }

    /// One input carrying every feature concatenated.
    ///
    /// The width is the sum of the configured features' widths, and must equal the width
    /// the model declares. Each feature is paired with its transformation here, so a
    /// request resolves none.
    ///
    /// # Errors
    ///
    /// Fails if the model declares more than one input, if its element type is not the
    /// configured `data_type`, or if the configured features produce no values or a width
    /// the model does not take.
    fn vectorised(
        mut transformations: HashMap<String, Box<dyn Transformation>>,
        feature_order: Vec<String>,
        data_type: crate::config::vectorization_config::VectorType,
        specs: &[IoSpec],
    ) -> Result<Self, InferenceError> {
        if specs.len() != 1 {
            let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
            return Err(format!(
                "this configuration concatenates every feature into one input, but the \
                 model declares {}: {}. Use \"model_inputs\" to feed each input by name.",
                specs.len(),
                names.join(", ")
            )
            .into());
        }
        let spec = &specs[0];
        let element = data_type.element_type();
        if spec.element_type != element {
            return Err(format!(
                "\"data_type\": {:?} makes the batch {}, but model input {:?} is {}",
                data_type_name(data_type),
                element,
                spec.name,
                spec.element_type
            )
            .into());
        }

        let width =
            crate::config::vectorization_config::vector_width(&transformations, &feature_order)?;
        if width == 0 {
            return Err(
                "the configured features produce no values, so the model's input \
                 would be empty"
                    .into(),
            );
        }
        if let Some(declared) = spec.width
            && declared != width
        {
            return Err(format!(
                "model input {:?} takes {declared} values per row, but the configured \
                 features produce {width}",
                spec.name
            )
            .into());
        }

        let mut plan = Vec::with_capacity(feature_order.len());
        for feature in feature_order {
            let transformation =
                transformations
                    .remove(&feature)
                    .ok_or_else(|| -> InferenceError {
                        format!(
                            "feature {feature:?} has no transformation, which startup should \
                     have caught"
                        )
                        .into()
                    })?;
            plan.push((feature, transformation));
        }

        Ok(Self {
            inputs: vec![PlannedInput {
                name: spec.name.clone(),
                element,
                width,
                source: Source::Vectorised(plan),
            }],
        })
    }

    /// One input per named feature, each read from the request verbatim.
    ///
    /// The declaration is checked both ways: every model input must be declared, and
    /// every declared input must exist in the model.
    ///
    /// Nothing is transformed here, so a width comes from the model alone and a row that
    /// omits a feature is an error rather than a defaulted zero.
    ///
    /// # Errors
    ///
    /// Fails if an input is undeclared, if a declared input is not in the model, if a
    /// declared type is not the model's, or if a scalar is declared for a wider input.
    fn named(
        model_inputs: std::collections::BTreeMap<String, InputDataType>,
        specs: &[IoSpec],
    ) -> Result<Self, InferenceError> {
        for spec in specs {
            if !model_inputs.contains_key(&spec.name) {
                return Err(format!(
                    "model input {:?} is not declared in model_inputs, so nothing would \
                     feed it.{} {} input(s) are declared.",
                    spec.name,
                    did_you_mean(&spec.name, model_inputs.keys().map(String::as_str)),
                    model_inputs.len()
                )
                .into());
            }
        }
        for name in model_inputs.keys() {
            if !specs.iter().any(|s| &s.name == name) {
                return Err(format!(
                    "model_inputs declares {name:?}, but the model has no such input.{} \
                     The model takes {} input(s).",
                    did_you_mean(name, specs.iter().map(|s| s.name.as_str())),
                    specs.len()
                )
                .into());
            }
        }

        let mut inputs = Vec::with_capacity(specs.len());
        for spec in specs {
            let declared = model_inputs[&spec.name];
            if declared.element_type() != spec.element_type {
                return Err(format!(
                    "model input {:?} is declared {declared} in model_inputs, which is \
                     {}, but the model expects {}",
                    spec.name,
                    declared.element_type(),
                    spec.element_type
                )
                .into());
            }

            let width = required_width(spec)?;

            if !declared.is_array() && width != 1 {
                return Err(format!(
                    "model input {:?} is declared {declared}, which is one value per \
                     row, but the model takes {width} values per row for it. Declare it \
                     as {}_array.",
                    spec.name,
                    declared.name()
                )
                .into());
            }
            inputs.push(PlannedInput {
                name: spec.name.clone(),
                element: spec.element_type,
                width,
                source: Source::Feature(spec.name.clone()),
            });
        }

        Ok(Self { inputs })
    }

    /// Builds one batch from every row the request carries, unpadded.
    ///
    /// Test-only; the serving path knows its chunk and its batch height and goes through
    /// [`InputBuilder::build_padded`].
    #[cfg(test)]
    pub(crate) fn build(&self, columns: &RequestColumns) -> Result<InputBatch, InferenceError> {
        let rows = columns.rows();
        self.build_padded(columns, 0..rows, rows)
    }

    /// Builds one batch of model inputs from the rows in `chunk`, `pad_to` rows tall.
    ///
    /// A concatenated input transforms its features; a named one is read as sent. The
    /// element type then only chooses which buffer to fill. Nothing is hashed here: the
    /// columns were indexed once when the request arrived.
    ///
    /// `pad_to` is at least the chunk's height, and any excess is **ballast**: a pinned
    /// model wants a fixed batch height, so the rows past the real ones are scored and
    /// thrown away without ever reaching a response or a log. Ballast is written at the
    /// end of each buffer -- zeros, or empty text, which no caller can observe.
    ///
    /// # Errors
    ///
    /// Fails if a feature a named input needs is absent or is the wrong element type, if
    /// a column does not carry `rows × width` values, or if a transformation refuses a
    /// value.
    pub(crate) fn build_padded(
        &self,
        columns: &RequestColumns,
        chunk: Range<usize>,
        pad_to: usize,
    ) -> Result<InputBatch, InferenceError> {
        let height = chunk.end.saturating_sub(chunk.start);
        debug_assert!(pad_to >= height, "padding cannot drop rows");
        let pad_to = pad_to.max(height);
        let mut batch = InputBatch::new(pad_to);

        for input in &self.inputs {
            let data = match &input.source {
                Source::Vectorised(plan) => {
                    let bound = bind(plan, columns)?;
                    match input.element {
                        ElementType::F32 => FeatureData::F32(build_vectorised_f32(
                            input,
                            plan,
                            &bound,
                            columns,
                            chunk.clone(),
                            pad_to,
                        )?),
                        ElementType::F64 => FeatureData::F64(build_vectorised_f64(
                            input,
                            plan,
                            &bound,
                            columns,
                            chunk.clone(),
                            pad_to,
                        )?),
                        other => {
                            return Err(format!(
                                "model input {:?} is {other} and cannot be built by \
                                 concatenating transformations, which produce floats",
                                input.name
                            )
                            .into());
                        }
                    }
                }
                Source::Feature(feature) => {
                    let column = columns
                        .get(feature)
                        .ok_or_else(|| missing_feature(input, feature, columns.names()))?;
                    check_column(input, feature, column, columns.rows())?;
                    let src = chunk.start * input.width..chunk.end * input.width;
                    match input.element {
                        ElementType::F32 => {
                            FeatureData::F32(verbatim_f32(input, feature, column, src, pad_to)?)
                        }
                        ElementType::F64 => {
                            FeatureData::F64(verbatim_f64(input, feature, column, src, pad_to)?)
                        }
                        ElementType::I32 => {
                            FeatureData::I32(verbatim_i32(input, feature, column, src, pad_to)?)
                        }
                        ElementType::I64 => {
                            FeatureData::I64(verbatim_i64(input, feature, column, src, pad_to)?)
                        }
                        ElementType::Str => {
                            FeatureData::Str(verbatim_str(input, feature, column, src, pad_to)?)
                        }
                    }
                }
            };
            batch.push(input.name.clone(), data)?;
        }
        Ok(batch)
    }

    /// How each model input is fed, for the startup banner.
    ///
    /// A concatenated input names its features in full up to eight of them, and is
    /// summarised beyond that.
    pub(crate) fn describe(&self) -> String {
        self.inputs
            .iter()
            .map(|input| {
                let how = match &input.source {
                    Source::Vectorised(plan) => {
                        const NAMED: usize = 8;
                        let names: Vec<&str> = plan.iter().map(|(name, _)| name.as_str()).collect();
                        if names.len() <= NAMED {
                            format!("[{}] transformed and concatenated", names.join(", "))
                        } else {
                            format!(
                                "{} features transformed and concatenated [{}, ... +{} more]",
                                names.len(),
                                names[..NAMED].join(", "),
                                names.len() - NAMED,
                            )
                        }
                    }
                    Source::Feature(name) => format!("feature {name:?}, verbatim"),
                };
                format!("{} {}[{}] <- {how}", input.name, input.element, input.width)
            })
            .collect::<Vec<_>>()
            .join("\n             ")
    }
}

/// Each planned feature paired with the column that feeds it, resolved once per chunk.
///
/// `None` means the request left the feature out, which a transformation covers with its
/// default. The width travels with the column because slicing a row out of it needs both.
type Bound<'a> = Vec<(
    &'a str,
    Option<(&'a FeatureData, usize)>,
    &'a dyn Transformation,
)>;

/// Pairs every planned feature with its column, and settles each one's width.
///
/// # Errors
///
/// Fails if a column's length is not a whole number of rows.
fn bind<'a>(
    plan: &'a [(String, Box<dyn Transformation>)],
    columns: &'a RequestColumns,
) -> Result<Bound<'a>, InferenceError> {
    let rows = columns.rows();
    let mut bound = Vec::with_capacity(plan.len());
    for (name, transformation) in plan {
        let found = match columns.get(name) {
            None => None,
            Some(data) if rows == 0 => Some((data, 0)),
            Some(data) => {
                if !data.len().is_multiple_of(rows) {
                    return Err(format!(
                        "feature {name:?} carries {} values, which is not a whole number \
                         of rows for a request of {rows}",
                        data.len()
                    )
                    .into());
                }
                Some((data, data.len() / rows))
            }
        };
        bound.push((name.as_str(), found, transformation.as_ref()));
    }
    Ok(bound)
}

/// The one concatenated float input: every feature transformed, in order.
///
/// Each transformation appends straight into the buffer being built, and an absent
/// feature contributes its default. The tail up to `pad_to` rows is ballast: one row of
/// every transformation's default, repeated, which is what an absent feature produces.
///
/// # Errors
///
/// Fails if a column runs short of the row asked for, if a transformation refuses a
/// value, or if a row does not produce the input's width.
fn build_vectorised_f32(
    input: &PlannedInput,
    plan: &[(String, Box<dyn Transformation>)],
    bound: &Bound<'_>,
    columns: &RequestColumns,
    chunk: Range<usize>,
    pad_to: usize,
) -> Result<Vec<f32>, InferenceError> {
    let mut values = Vec::with_capacity(pad_to * input.width);
    for row in chunk {
        let before = values.len();
        for (name, column, transformation) in bound {
            match column {
                Some((data, width)) => {
                    let slice = data
                        .row(row, *width)
                        .ok_or_else(|| short_column(name, columns.row_id(row), row))?;
                    transformation.transform(slice, &mut values).map_err(
                        |e| -> InferenceError {
                            format!("feature {name:?} in row {:?}: {e}", columns.row_id(row)).into()
                        },
                    )?;
                }
                None => values.extend_from_slice(transformation.get_default_val()),
            }
        }
        check_row(input, columns.row_id(row), values.len() - before)?;
    }
    pad_with(&mut values, pad_to * input.width, &default_row_f32(plan));
    Ok(values)
}

/// The same, in double precision.
fn build_vectorised_f64(
    input: &PlannedInput,
    plan: &[(String, Box<dyn Transformation>)],
    bound: &Bound<'_>,
    columns: &RequestColumns,
    chunk: Range<usize>,
    pad_to: usize,
) -> Result<Vec<f64>, InferenceError> {
    let mut values = Vec::with_capacity(pad_to * input.width);
    for row in chunk {
        let before = values.len();
        for (name, column, transformation) in bound {
            match column {
                Some((data, width)) => {
                    let slice = data
                        .row(row, *width)
                        .ok_or_else(|| short_column(name, columns.row_id(row), row))?;
                    transformation.transform_f64(slice, &mut values).map_err(
                        |e| -> InferenceError {
                            format!("feature {name:?} in row {:?}: {e}", columns.row_id(row)).into()
                        },
                    )?;
                }
                None => values.extend(transformation.default_val_f64()),
            }
        }
        check_row(input, columns.row_id(row), values.len() - before)?;
    }
    pad_with(&mut values, pad_to * input.width, &default_row_f64(plan));
    Ok(values)
}

/// The request's feature columns, in the service's own types.
///
/// The feature map is decoded once, when the request arrives. Names are hashed once here
/// rather than once per row per feature.
///
/// # Fields
///
/// * `row_ids` - one id per row, which states the row count
/// * `columns` - one entry per feature sent, keyed by name
#[derive(Debug)]
pub(crate) struct RequestColumns {
    row_ids: Vec<String>,
    columns: HashMap<String, FeatureData>,
}

impl RequestColumns {
    /// Decodes a request's feature map into the service's own columns.
    ///
    /// The row count is `row_ids.len()`, stated once for the whole request, and every
    /// column is checked to hold a whole number of rows. Each feature's width comes from
    /// the model, so the widths themselves are checked later, against that.
    ///
    /// A name cannot arrive twice: the wire is a map, so protobuf decoding settles that
    /// before the service sees it.
    ///
    /// # Errors
    ///
    /// Fails on anything [`proto_columns::decode`] refuses.
    pub(crate) fn from_request(
        row_ids: Vec<String>,
        features: HashMap<String, Feature>,
    ) -> Result<Self, InferenceError> {
        let columns = proto_columns::decode(features, row_ids.len())?;
        Ok(Self { row_ids, columns })
    }

    /// Rows this request carries, which every feature's width is measured against.
    pub(crate) fn rows(&self) -> usize {
        self.row_ids.len()
    }

    pub(crate) fn row_ids(&self) -> &[String] {
        &self.row_ids
    }

    /// The id of one row, for an error that has to say which row it means.
    fn row_id(&self, row: usize) -> &str {
        self.row_ids.get(row).map_or("<unknown>", String::as_str)
    }

    fn get(&self, name: &str) -> Option<&FeatureData> {
        self.columns.get(name)
    }

    /// Every feature name sent, for a near-miss suggestion.
    fn names(&self) -> impl Iterator<Item = &str> {
        self.columns.keys().map(String::as_str)
    }

    /// Gives the ids and the features back, for the inference log.
    ///
    /// The columns are moved back into the wire's own shape, so the log records the named
    /// values the caller sent -- before any transformation -- and no feature data is
    /// copied to do it. Each width is restated from the row count, so a reader sees the
    /// shape the caller sent as well as its values.
    pub(crate) fn into_wire(self) -> (Vec<String>, HashMap<String, Feature>) {
        let rows = self.row_ids.len();
        (self.row_ids, proto_columns::encode(self.columns, rows))
    }
}

/// Checks that a column carries `rows × width` values for the input it feeds.
///
/// Checked once per input rather than once per row: a ragged feature is a property of the
/// request, not of a row inside it.
///
/// # Errors
///
/// Fails when the count does not match. The message names the width the count implies,
/// since which row is wrong cannot be known -- the values are positional, so a short
/// column gives no clue where the gap was meant to be.
fn check_column(
    input: &PlannedInput,
    feature: &str,
    column: &FeatureData,
    rows: usize,
) -> Result<(), InferenceError> {
    let wanted = rows * input.width;
    if column.len() == wanted {
        return Ok(());
    }
    let implied = if rows > 0 && column.len().is_multiple_of(rows) {
        format!(", which is {} per row", column.len() / rows)
    } else {
        format!(", which is not a whole number of rows for {rows}")
    };
    Err(format!(
        "model input {:?} takes {} values per row, so feature {feature:?} must carry \
         {wanted} for {rows} row(s), but it carries {}{implied}",
        input.name,
        input.width,
        column.len()
    )
    .into())
}

/// A 32-bit float input, read from the request verbatim.
///
/// A `float` column is copied and an `integer` column widens, which is exact. A `double`,
/// a `long` or text is refused, because the conversion could lose precision. The tail up
/// to `pad_to` rows is zero ballast.
fn verbatim_f32(
    input: &PlannedInput,
    feature: &str,
    column: &FeatureData,
    src: Range<usize>,
    pad_to: usize,
) -> Result<Vec<f32>, InferenceError> {
    let mut values = Vec::with_capacity(pad_to * input.width);
    match column {
        FeatureData::F32(v) => values.extend_from_slice(&v[src]),
        FeatureData::I32(v) => values.extend(v[src].iter().map(|x| *x as f32)),
        other => return Err(wrong_type(input, feature, "a float or an integer", other)),
    }
    values.resize(pad_to * input.width, 0.0);
    Ok(values)
}

/// A 64-bit float input, read from the request verbatim.
///
/// A `float` and an `integer` both widen to a double exactly, so both are accepted. A
/// `long` is not: past 2^53 it would not survive the conversion. The tail up to `pad_to`
/// rows is zero ballast.
fn verbatim_f64(
    input: &PlannedInput,
    feature: &str,
    column: &FeatureData,
    src: Range<usize>,
    pad_to: usize,
) -> Result<Vec<f64>, InferenceError> {
    let mut values = Vec::with_capacity(pad_to * input.width);
    match column {
        FeatureData::F64(v) => values.extend_from_slice(&v[src]),
        FeatureData::F32(v) => values.extend(v[src].iter().copied().map(f64::from)),
        FeatureData::I32(v) => values.extend(v[src].iter().copied().map(f64::from)),
        other => {
            return Err(wrong_type(
                input,
                feature,
                "a double, a float or an integer",
                other,
            ));
        }
    }
    values.resize(pad_to * input.width, 0.0);
    Ok(values)
}

/// A 32-bit integer input, read from the request verbatim.
///
/// An `integer` column only. A `long` is **not** accepted, because narrowing it could
/// overflow. Widening the other way is exact, which is why [`verbatim_i64`] does accept
/// an integer. The tail up to `pad_to` rows is zero ballast.
fn verbatim_i32(
    input: &PlannedInput,
    feature: &str,
    column: &FeatureData,
    src: Range<usize>,
    pad_to: usize,
) -> Result<Vec<i32>, InferenceError> {
    let mut values = Vec::with_capacity(pad_to * input.width);
    match column {
        FeatureData::I32(v) => values.extend_from_slice(&v[src]),
        other => return Err(wrong_type(input, feature, "an integer", other)),
    }
    values.resize(pad_to * input.width, 0);
    Ok(values)
}

/// A 64-bit integer input, read from the request verbatim.
///
/// An `integer` column is accepted as well as a `long`, because widening it is exact. The
/// tail up to `pad_to` rows is zero ballast.
fn verbatim_i64(
    input: &PlannedInput,
    feature: &str,
    column: &FeatureData,
    src: Range<usize>,
    pad_to: usize,
) -> Result<Vec<i64>, InferenceError> {
    let mut values = Vec::with_capacity(pad_to * input.width);
    match column {
        FeatureData::I64(v) => values.extend_from_slice(&v[src]),
        FeatureData::I32(v) => values.extend(v[src].iter().copied().map(i64::from)),
        other => return Err(wrong_type(input, feature, "a long or an integer", other)),
    }
    values.resize(pad_to * input.width, 0);
    Ok(values)
}

/// A text input, read from the request verbatim.
///
/// A `string` column only. The tail up to `pad_to` rows is empty-string ballast.
fn verbatim_str(
    input: &PlannedInput,
    feature: &str,
    column: &FeatureData,
    src: Range<usize>,
    pad_to: usize,
) -> Result<Vec<String>, InferenceError> {
    let mut values = Vec::with_capacity(pad_to * input.width);
    match column {
        FeatureData::Str(v) => values.extend(v[src].iter().cloned()),
        other => return Err(wrong_type(input, feature, "text", other)),
    }
    values.resize(pad_to * input.width, String::new());
    Ok(values)
}

/// A feature the request did not send at all.
///
/// A named input has no transformation to supply a default, and a dense tensor has no way
/// to say "absent", so this is an error. A feature is present for the whole request or
/// absent from it, so the message names the input and the feature rather than a row.
fn missing_feature<'a>(
    input: &PlannedInput,
    feature: &str,
    sent: impl Iterator<Item = &'a str>,
) -> InferenceError {
    format!(
        "model input {:?} is fed verbatim from feature {feature:?}, which the request \
         does not send.{} Send it, or configure \"feature_order\" so a transformation can \
         supply a default.",
        input.name,
        did_you_mean(feature, sent)
    )
    .into()
}

/// A feature the request sent as the wrong type for its input.
fn wrong_type(
    input: &PlannedInput,
    feature: &str,
    wanted: &str,
    got: &FeatureData,
) -> InferenceError {
    format!(
        "model input {:?} is {} and is fed verbatim, so feature {feature:?} must be \
         {wanted}, but the request sent {}",
        input.name,
        input.element,
        got.element_type()
    )
    .into()
}

/// A column that ran out before the row a transformation asked for.
fn short_column(feature: &str, row_id: &str, row: usize) -> InferenceError {
    format!("feature {feature:?} has no values for row {row_id:?}, which is row {row} of the batch")
        .into()
}

/// Repeats `unit` onto `values` until it is `target` long.
///
/// `unit` is one row's worth, so the tail lands on a row boundary exactly.
fn pad_with<T: Clone>(values: &mut Vec<T>, target: usize, unit: &[T]) {
    if unit.is_empty() {
        return;
    }
    while values.len() < target {
        let room = target - values.len();
        let take = room.min(unit.len());
        values.extend_from_slice(&unit[..take]);
    }
}

/// One ballast row for a concatenated float input: every transformation's default.
fn default_row_f32(plan: &[(String, Box<dyn Transformation>)]) -> Vec<f32> {
    plan.iter()
        .flat_map(|(_, t)| t.get_default_val().iter().copied())
        .collect()
}

/// The same, in double precision.
fn default_row_f64(plan: &[(String, Box<dyn Transformation>)]) -> Vec<f64> {
    plan.iter().flat_map(|(_, t)| t.default_val_f64()).collect()
}

/// Checks that a row produced the width settled at startup.
///
/// # Errors
///
/// Fails when it did not: a batch whose rows differ in width has no rectangular shape.
fn check_row(input: &PlannedInput, row_id: &str, produced: usize) -> Result<(), InferenceError> {
    if produced == input.width {
        return Ok(());
    }
    Err(format!(
        "model input {:?} takes {} values per row, but row {row_id:?} produced {produced}",
        input.name, input.width
    )
    .into())
}

/// The nearest name to `wanted`, phrased for an error, or nothing if none is close.
///
/// Ranked by what a typo leaves intact: the shared prefix plus the shared suffix,
/// counting no character from both ends. A candidate must share more than half of
/// `wanted` to be offered, so a genuinely absent name suggests nothing rather than
/// something misleading.
fn did_you_mean<'a>(wanted: &str, candidates: impl Iterator<Item = &'a str>) -> String {
    let overlap = |other: &str| {
        let front = wanted
            .bytes()
            .zip(other.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let back = wanted
            .bytes()
            .rev()
            .zip(other.bytes().rev())
            .take_while(|(a, b)| a == b)
            .count();
        (front + back).min(wanted.len()).min(other.len())
    };

    candidates
        .map(|name| (overlap(name), name))
        .filter(|(score, _)| *score * 2 > wanted.len())
        .max_by_key(|(score, _)| *score)
        .map(|(_, name)| format!(" Did you mean {name:?}?"))
        .unwrap_or_default()
}

/// Values per row for one input, as the model declares it.
///
/// # Errors
///
/// Fails when the model leaves the axis dynamic, since a row read verbatim would then
/// have no shape to fill.
fn required_width(spec: &IoSpec) -> Result<usize, InferenceError> {
    spec.width.ok_or_else(|| {
        format!(
            "model input {:?} leaves its width dynamic, so a row read verbatim would \
             have no shape to fill. Export the model with a fixed width for this input, \
             or use \"feature_order\" so a transformation settles the width.",
            spec.name
        )
        .into()
    })
}

/// The configuration spelling of a vector's element type, for an error.
fn data_type_name(data_type: crate::config::vectorization_config::VectorType) -> &'static str {
    match data_type {
        crate::config::vectorization_config::VectorType::Float => "float",
        crate::config::vectorization_config::VectorType::Double => "double",
    }
}

/// Building requests the way a test describes them: row by row.
///
/// The wire is columnar, so this transposes rows into columns, builds the feature map and
/// goes through [`RequestColumns::from_request`], the same decode the service uses.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{FeatureData, RequestColumns};
    use crate::inference::proto_columns::test_support::{Column, features};

    /// One feature's value for one row.
    ///
    /// Held as a vector because a feature may be several values wide, and because a
    /// transformation reads a slice that has to be borrowed from somewhere.
    #[derive(Clone, Debug)]
    pub(crate) enum V {
        F32(Vec<f32>),
        F64(Vec<f64>),
        I32(Vec<i32>),
        I64(Vec<i64>),
        Str(Vec<String>),
    }

    impl V {
        /// One 32-bit float.
        pub(crate) fn f32(v: f32) -> Self {
            V::F32(vec![v])
        }
        pub(crate) fn f64(v: f64) -> Self {
            V::F64(vec![v])
        }
        pub(crate) fn i32(v: i32) -> Self {
            V::I32(vec![v])
        }
        pub(crate) fn i64(v: i64) -> Self {
            V::I64(vec![v])
        }
        pub(crate) fn str(v: &str) -> Self {
            V::Str(vec![v.to_owned()])
        }

        /// Values this holds, which for one row's value is that feature's width.
        pub(crate) fn len(&self) -> usize {
            match self {
                V::F32(v) => v.len(),
                V::F64(v) => v.len(),
                V::I32(v) => v.len(),
                V::I64(v) => v.len(),
                V::Str(v) => v.len(),
            }
        }

        /// Appends this value onto a column of the same type.
        fn extend(&self, into: &mut V) {
            match (self, into) {
                (V::F32(from), V::F32(to)) => to.extend_from_slice(from),
                (V::F64(from), V::F64(to)) => to.extend_from_slice(from),
                (V::I32(from), V::I32(to)) => to.extend_from_slice(from),
                (V::I64(from), V::I64(to)) => to.extend_from_slice(from),
                (V::Str(from), V::Str(to)) => to.extend(from.iter().cloned()),
                (from, to) => {
                    panic!("test feature changes type between rows: {from:?} then {to:?}")
                }
            }
        }

        fn into_column(self) -> FeatureData {
            match self {
                V::F32(v) => FeatureData::F32(v),
                V::F64(v) => FeatureData::F64(v),
                V::I32(v) => FeatureData::I32(v),
                V::I64(v) => FeatureData::I64(v),
                V::Str(v) => FeatureData::Str(v),
            }
        }
    }

    /// One row: its id, and the features it sets.
    pub(crate) type Row<'a> = (&'a str, Vec<(&'a str, V)>);

    /// One row, for a test that needs exactly one.
    pub(crate) fn row<'a>(row_id: &'a str, features: &[(&'a str, V)]) -> Row<'a> {
        (row_id, features.to_vec())
    }

    /// Transposes rows into the columnar request the service takes.
    ///
    /// A feature no row sets is absent from the request, which is how a default is
    /// exercised. A feature only *some* rows set is refused here rather than encoded: one
    /// entry holds the whole batch, so there is no way to express a gap in it.
    ///
    /// # Panics
    ///
    /// Panics if a feature changes element type or width between rows, or if some rows
    /// set it and others do not.
    pub(crate) fn request(rows: &[Row<'_>]) -> RequestColumns {
        let row_ids: Vec<String> = rows.iter().map(|(id, _)| (*id).to_owned()).collect();
        let mut order: Vec<&str> = Vec::new();
        for (_, features) in rows {
            for (name, _) in features {
                if !order.contains(name) {
                    order.push(name);
                }
            }
        }
        let mut columns = Vec::with_capacity(order.len());
        for name in order {
            let mut column: Option<V> = None;
            let mut width = 0;
            for (_, row_features) in rows {
                let value = row_features
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, value)| value)
                    .unwrap_or_else(|| {
                        panic!(
                            "test feature {name:?} is set by some rows and not others, which \
                             one column cannot carry; leave it out of every row instead"
                        )
                    });
                match &mut column {
                    None => {
                        width = value.len();
                        column = Some(value.clone());
                    }
                    Some(into) => {
                        assert_eq!(
                            value.len(),
                            width,
                            "test feature {name:?} changes width between rows"
                        );
                        value.extend(into);
                    }
                }
            }
            columns.push((
                name.to_owned(),
                column.expect("a listed feature has a value").into_column(),
                width,
            ));
        }
        build(row_ids, columns)
    }

    /// A request built from whole columns, for shapes rows cannot express -- a column
    /// whose width disagrees with the model's, say.
    ///
    /// Each entry is a name, its values per row, and every row's values back to back.
    pub(crate) fn columns(row_ids: &[&str], features: Vec<(&str, usize, V)>) -> RequestColumns {
        let ids = row_ids.iter().map(|id| (*id).to_owned()).collect();
        let columns: Vec<Column> = features
            .into_iter()
            .map(|(name, width, values)| (name.to_owned(), values.into_column(), width))
            .collect();
        build(ids, columns)
    }

    /// Builds the feature map and decodes it back, so a test goes through the wire.
    ///
    /// Each column states its own width, exactly as a caller would, so a test request is
    /// held to the same self-consistency check as a real one.
    fn build(row_ids: Vec<String>, columns: Vec<Column>) -> RequestColumns {
        RequestColumns::from_request(row_ids, features(columns))
            .expect("a well-formed test request")
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{V, request, row};
    use super::*;

    fn spec(name: &str, element_type: ElementType, width: Option<usize>) -> IoSpec {
        IoSpec::new(name, element_type, width)
    }

    fn config(json: &str) -> VectorizationConfig {
        VectorizationConfig::from_json(json).expect("valid configuration")
    }

    /// Concatenation follows the configuration's order, not the order the request sends.
    #[test]
    fn a_vectorised_config_concatenates_in_the_configured_order() {
        let builder = InputBuilder::resolve(
            config(
                r#"{
                    "feature_transformations": {
                        "a": {"type": "identity", "default_val": [0.0]},
                        "b": {"type": "identity", "default_val": [0.0]}
                    },
                    "feature_order": ["a", "b"]
                }"#,
            ),
            &[spec("features", ElementType::F32, Some(2))],
        )
        .expect("2 = 1 + 1");

        let batch = builder
            .build(&request(&[
                row("r1", &[("a", V::f32(1.0)), ("b", V::f32(2.0))]),
                row("r2", &[("b", V::f32(4.0)), ("a", V::f32(3.0))]),
            ]))
            .expect("both rows are complete");

        assert_eq!(batch.rows(), 2);
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0])
        );
    }

    /// A double vector keeps 0.1 exactly, which the f32 path would not.
    #[test]
    fn a_double_vector_keeps_its_precision() {
        let builder = InputBuilder::resolve(
            config(
                r#"{
                    "data_type": "double",
                    "feature_transformations": {"a": {"type": "identity", "default_val": [0.0]}},
                    "feature_order": ["a"]
                }"#,
            ),
            &[spec("features", ElementType::F64, Some(1))],
        )
        .expect("one double input");

        let batch = builder
            .build(&request(&[row("r1", &[("a", V::f64(0.1))])]))
            .expect("one row");
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F64(vec![0.1])
        );
    }

    #[test]
    fn a_vector_of_the_wrong_width_is_refused_with_both_numbers() {
        let err = InputBuilder::resolve(
            config(r#"{"feature_order": ["a", "b"]}"#),
            &[spec("features", ElementType::F32, Some(8))],
        )
        .expect_err("8 != 2");
        let m = err.to_string();
        assert!(m.contains('8') && m.contains('2'), "got {m}");
    }

    #[test]
    fn a_vectorised_config_against_a_multi_input_model_is_refused() {
        let err = InputBuilder::resolve(
            config(r#"{"feature_order": ["a"]}"#),
            &[
                spec("numbers", ElementType::F32, Some(1)),
                spec("tags", ElementType::Str, Some(1)),
            ],
        )
        .expect_err("one vector cannot feed two inputs");
        let m = err.to_string();
        assert!(
            m.contains("model_inputs") && m.contains("tags"),
            "the error must point at the other shape: {m}"
        );
    }

    #[test]
    fn a_float_config_against_a_double_model_is_refused() {
        let err = InputBuilder::resolve(
            config(r#"{"data_type": "float", "feature_order": ["a"]}"#),
            &[spec("features", ElementType::F64, Some(1))],
        )
        .expect_err("FP32 is not FP64");
        let m = err.to_string();
        assert!(m.contains("FP32") && m.contains("FP64"), "got {m}");
    }

    /// Ballast is each transformation's default for a concatenated input, and zero for a
    /// verbatim one.
    #[test]
    fn ballast_repeats_what_a_featureless_row_would_have_produced() {
        let vectorised = InputBuilder::resolve(
            config(
                r#"{
                    "feature_transformations": {
                        "age": {"type": "identity", "default_val": [7.0]}
                    },
                    "feature_order": ["age"]
                }"#,
            ),
            &[spec("features", ElementType::F32, Some(1))],
        )
        .expect("one concatenated input")
        .build_padded(&request(&[row("r1", &[("age", V::f32(1.0))])]), 0..1, 3)
        .expect("one real row, two of ballast");
        assert_eq!(
            vectorised.get("features").expect("features").data,
            FeatureData::F32(vec![1.0, 7.0, 7.0]),
            "ballast is the transformation's default, not a zero"
        );
        assert_eq!(
            vectorised.rows(),
            3,
            "the engine is given the padded height"
        );

        let verbatim = InputBuilder::resolve(
            config(r#"{"model_inputs": {"age": {"data_type": "float"}}}"#),
            &[spec("age", ElementType::F32, Some(1))],
        )
        .expect("one named input")
        .build_padded(&request(&[row("r1", &[("age", V::f32(1.0))])]), 0..1, 3)
        .expect("padding must not need features");
        assert_eq!(
            verbatim.get("age").expect("age").data,
            FeatureData::F32(vec![1.0, 0.0, 0.0]),
            "a verbatim input has no default, so its ballast is zero"
        );
    }

    /// A 3-wide input pads in whole rows of three, so no later row is read shifted.
    #[test]
    fn ballast_lands_on_a_row_boundary_for_a_wide_input() {
        let batch = InputBuilder::resolve(
            config(r#"{"model_inputs": {"v": {"data_type": "float_array"}}}"#),
            &[spec("v", ElementType::F32, Some(3))],
        )
        .expect("one array input")
        .build_padded(
            &request(&[row("r1", &[("v", V::F32(vec![1.0, 2.0, 3.0]))])]),
            0..1,
            3,
        )
        .expect("one real row, two of ballast");
        assert_eq!(
            batch.get("v").expect("v").data,
            FeatureData::F32(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            batch.get("v").expect("v").width(),
            3,
            "width survives padding"
        );
    }

    /// A transformation that refuses a value names the row it happened on, and no other.
    #[test]
    fn a_transformation_failure_names_the_row_it_happened_on() {
        let builder = InputBuilder::resolve(
            config(
                r#"{
                    "data_type": "float",
                    "feature_transformations": {
                        "n": {"type": "identity", "default_val": [0.0]}
                    },
                    "feature_order": ["n"]
                }"#,
            ),
            &[spec("features", ElementType::F32, Some(1))],
        )
        .expect("one concatenated input");

        let err = builder
            .build(&request(&[
                row("candidate-aaa", &[("n", V::str("1.5"))]),
                row("candidate-bbb", &[("n", V::str("not-a-number"))]),
            ]))
            .expect_err("the second row's text is not a number");
        let m = err.to_string();
        assert!(
            m.contains("candidate-bbb") && m.contains('n'),
            "the error must name the row and the feature: {m}"
        );
        assert!(
            !m.contains("candidate-aaa"),
            "and not the row that was fine: {m}"
        );
    }

    /// A ragged column reports the width its count implies, since the values are
    /// positional and cannot say which row is wrong.
    #[test]
    fn a_ragged_column_reports_the_width_its_count_implies() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"v": {"data_type": "float_array"}}}"#),
            &[spec("v", ElementType::F32, Some(3))],
        )
        .expect("one 3-wide input");

        let err = builder
            .build(&request(&[
                row("r1", &[("v", V::F32(vec![1.0, 2.0]))]),
                row("r2", &[("v", V::F32(vec![3.0, 4.0]))]),
            ]))
            .expect_err("2 wide is not 3 wide");
        let m = err.to_string();
        assert!(
            m.contains("which is 2 per row") && m.contains('3'),
            "the error should say the width sent and the width needed: {m}"
        );
    }

    /// `row_ids` states the row count, and every feature must carry `rows × width`, the
    /// width being the model's.
    #[test]
    fn row_ids_state_the_row_count() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"v": {"data_type": "float_array"}}}"#),
            &[spec("v", ElementType::F32, Some(2))],
        )
        .expect("one 2-wide input");

        let wide = |values: Vec<f32>| vec![("v", 2usize, V::F32(values))];

        let columns = test_support::columns(
            &["r1", "r2", "r3"],
            wide(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        assert_eq!(columns.rows(), 3, "three ids means three rows");

        let batch = builder.build(&columns).expect("6 values is 3 rows of 2");
        assert_eq!(batch.rows(), 3);
        assert_eq!(
            batch.get("v").expect("v").data,
            FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );

        let err = builder
            .build(&test_support::columns(
                &["r1", "r2"],
                vec![("v", 3usize, V::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))],
            ))
            .expect_err("3 values per row is not the 2 the model takes");
        let m = err.to_string();
        assert!(
            m.contains("must carry 4") && m.contains("which is 3 per row"),
            "the error should measure the column against rows × width: {m}"
        );
    }

    /// The order features arrive in changes nothing; `feature_order` decides the layout.
    #[test]
    fn the_order_features_arrive_in_does_not_matter() {
        let builder = InputBuilder::resolve(
            config(
                r#"{
                    "data_type": "float",
                    "feature_transformations": {
                        "a": {"type": "identity", "default_val": [0.0]},
                        "b": {"type": "identity", "default_val": [0.0]}
                    },
                    "feature_order": ["a", "b"]
                }"#,
            ),
            &[spec("features", ElementType::F32, Some(2))],
        )
        .expect("two features into one 2-wide input");

        let a = ("a", 1usize, V::f32(1.0));
        let b = ("b", 1usize, V::f32(2.0));

        let in_order = builder
            .build(&test_support::columns(&["r1"], vec![a.clone(), b.clone()]))
            .expect("a then b");
        let reversed = builder
            .build(&test_support::columns(&["r1"], vec![b, a]))
            .expect("b then a");

        assert_eq!(
            in_order.get("features").expect("input").data,
            FeatureData::F32(vec![1.0, 2.0]),
            "the configuration's order decides the layout"
        );
        assert_eq!(
            in_order, reversed,
            "and the request's order changes nothing at all"
        );
    }

    /// A float input copies a float column and widens an integer one, which is exact.
    #[test]
    fn a_verbatim_float_widens_an_integer_column() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"n": {"data_type": "float"}}}"#),
            &[spec("n", ElementType::F32, Some(1))],
        )
        .expect("one float input");

        let floats = builder
            .build(&request(&[
                row("r1", &[("n", V::f32(1.5))]),
                row("r2", &[("n", V::f32(2.5))]),
            ]))
            .expect("a float column is copied");
        assert_eq!(
            floats.get("n").expect("n").data,
            FeatureData::F32(vec![1.5, 2.5])
        );

        let integers = builder
            .build(&request(&[
                row("r1", &[("n", V::i32(7))]),
                row("r2", &[("n", V::i32(8))]),
            ]))
            .expect("an integer column widens");
        assert_eq!(
            integers.get("n").expect("n").data,
            FeatureData::F32(vec![7.0, 8.0])
        );
    }

    /// A float input refuses a double, a long or text rather than narrow it.
    #[test]
    fn a_verbatim_float_refuses_a_value_that_would_lose_precision() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"n": {"data_type": "float"}}}"#),
            &[spec("n", ElementType::F32, Some(1))],
        )
        .expect("one float input");

        for value in [V::f64(1.5), V::i64(9_007_199_254_740_993), V::str("1.5")] {
            let err = builder
                .build(&request(&[row("r1", &[("n", value.clone())])]))
                .expect_err("a narrowing conversion is refused");
            let m = err.to_string();
            assert!(
                m.contains('n') && m.contains("verbatim"),
                "the refusal should name the input and say it is verbatim; got {m}"
            );
        }
    }

    #[test]
    fn named_inputs_are_fed_by_the_feature_of_the_same_name() {
        let builder = InputBuilder::resolve(
            config(
                r#"{
                    "model_inputs": {
                        "age": {"data_type": "float"},
                        "city": {"data_type": "string"}
                    }
                }"#,
            ),
            &[
                spec("age", ElementType::F32, Some(1)),
                spec("city", ElementType::Str, Some(1)),
            ],
        )
        .expect("both inputs are declared");

        let batch = builder
            .build(&request(&[row(
                "r1",
                &[("age", V::f32(50.0)), ("city", V::str("london"))],
            )]))
            .expect("one row");

        assert_eq!(
            batch.get("age").expect("age").data,
            FeatureData::F32(vec![50.0]),
            "a named float reaches the model as sent, unscaled"
        );
        assert_eq!(
            batch.get("city").expect("city").data,
            FeatureData::Str(vec!["london".into()]),
            "text is passed through untouched"
        );
    }

    #[test]
    fn an_array_input_takes_a_whole_vector_per_row() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"city": {"data_type": "float_array"}}}"#),
            &[spec("city", ElementType::F32, Some(3))],
        )
        .expect("three values per row");

        let batch = builder
            .build(&request(&[
                row("r1", &[("city", V::F32(vec![0.0, 1.0, 0.0]))]),
                row("r2", &[("city", V::F32(vec![1.0, 0.0, 0.0]))]),
            ]))
            .expect("each row sends its own vector");

        assert_eq!(
            batch.get("city").expect("city").data,
            FeatureData::F32(vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        );
        assert_eq!(batch.get("city").expect("city").width(), 3);
    }

    #[test]
    fn an_undeclared_model_input_is_refused_at_startup() {
        let err = InputBuilder::resolve(
            config(r#"{"model_inputs": {"age": {"data_type": "float"}}}"#),
            &[
                spec("age", ElementType::F32, Some(1)),
                spec("city", ElementType::Str, Some(1)),
            ],
        )
        .expect_err("city would be left unfed");
        assert!(err.to_string().contains("city"), "got {err}");
    }

    #[test]
    fn a_declared_input_the_model_does_not_have_is_refused() {
        let err = InputBuilder::resolve(
            config(
                r#"{"model_inputs": {"age": {"data_type": "float"}, "ghost": {"data_type": "float"}}}"#,
            ),
            &[spec("age", ElementType::F32, Some(1))],
        )
        .expect_err("a typo in an input name must not pass");
        assert!(err.to_string().contains("ghost"), "got {err}");
    }

    #[test]
    fn a_declared_type_that_disagrees_with_the_model_is_refused() {
        let err = InputBuilder::resolve(
            config(r#"{"model_inputs": {"tags": {"data_type": "float"}}}"#),
            &[spec("tags", ElementType::Str, Some(1))],
        )
        .expect_err("FP32 is not BYTES");
        let m = err.to_string();
        assert!(m.contains("FP32") && m.contains("STRING"), "got {m}");
    }

    #[test]
    fn a_scalar_declaration_against_a_wide_input_points_at_the_array_spelling() {
        let err = InputBuilder::resolve(
            config(r#"{"model_inputs": {"city": {"data_type": "float"}}}"#),
            &[spec("city", ElementType::F32, Some(3))],
        )
        .expect_err("one value per row cannot fill three");
        let m = err.to_string();
        assert!(m.contains("float_array"), "the fix must be named: {m}");
    }

    #[test]
    fn a_transformation_wider_than_its_input_is_refused_with_both_numbers() {
        let err = InputBuilder::resolve(
            config(
                r#"{
                    "feature_transformations": {
                        "city": {"type": "one_hot_encoding", "categories": ["a", "b", "c"], "default_val": [0, 0, 0]}
                    },
                    "feature_order": ["city"]
                }"#,
            ),
            &[spec("wide", ElementType::F32, Some(5))],
        )
        .expect_err("3 != 5");
        let m = err.to_string();
        assert!(m.contains('3') && m.contains('5'), "got {m}");
    }

    #[test]
    fn a_wrongly_typed_feature_for_a_verbatim_input_names_both_types() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"tags": {"data_type": "string"}}}"#),
            &[spec("tags", ElementType::Str, Some(1))],
        )
        .expect("one text input");

        let err = builder
            .build(&request(&[row("r7", &[("tags", V::f32(1.0))])]))
            .expect_err("a float is not text");
        let m = err.to_string();
        assert!(
            m.contains("tags") && m.contains("STRING") && m.contains("FP32"),
            "the error must name the input, the type it needs and what arrived: {m}"
        );
    }

    #[test]
    fn an_absent_text_feature_is_refused() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"tags": {"data_type": "string"}}}"#),
            &[spec("tags", ElementType::Str, Some(1))],
        )
        .expect("one text input");

        let err = builder
            .build(&request(&[row("r1", &[])]))
            .expect_err("an empty string is a value, not an absence");
        let m = err.to_string();
        assert!(
            m.contains("tags") && m.contains("does not send"),
            "the refusal should name the feature the request left out; got {m}"
        );
    }

    /// An absent feature for a verbatim input is refused, not defaulted to zero.
    #[test]
    fn an_absent_long_feature_is_refused() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"code": {"data_type": "long"}}}"#),
            &[spec("code", ElementType::I64, Some(1))],
        )
        .expect("one integer input");

        let err = builder
            .build(&request(&[row("r1", &[])]))
            .expect_err("a verbatim input cannot be defaulted");
        let m = err.to_string();
        assert!(
            m.contains("code") && m.contains("does not send"),
            "the refusal should name the feature the request left out; got {m}"
        );
    }

    #[test]
    fn an_integer_feature_widens_into_a_long_input() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"code": {"data_type": "long"}}}"#),
            &[spec("code", ElementType::I64, Some(1))],
        )
        .expect("one integer input");

        let batch = builder
            .build(&request(&[row("r1", &[("code", V::i32(7))])]))
            .expect("an integer is a long");
        assert_eq!(
            batch.get("code").expect("code").data,
            FeatureData::I64(vec![7])
        );
    }

    #[test]
    fn an_int_input_takes_a_32_bit_integer_feature() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"code": {"data_type": "int"}}}"#),
            &[spec("code", ElementType::I32, Some(1))],
        )
        .expect("one 32-bit integer input");

        let batch = builder
            .build(&request(&[
                row("r1", &[("code", V::i32(7))]),
                row("r2", &[("code", V::i32(0))]),
            ]))
            .expect("both rows send a code");
        assert_eq!(
            batch.get("code").expect("code").data,
            FeatureData::I32(vec![7, 0])
        );
    }

    /// A `long` column is refused for an `int` input, because narrowing could overflow.
    #[test]
    fn a_long_feature_is_refused_for_an_int_input() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"code": {"data_type": "int"}}}"#),
            &[spec("code", ElementType::I32, Some(1))],
        )
        .expect("one 32-bit integer input");

        let err = builder
            .build(&request(&[row(
                "r4",
                &[("code", V::i64(i64::from(i32::MAX) + 1))],
            )]))
            .expect_err("a long may not fit");
        let m = err.to_string();
        assert!(
            m.contains("code") && m.contains("INT32") && m.contains("INT64"),
            "the error must name the input, the type it needs and what arrived: {m}"
        );
    }

    #[test]
    fn an_int_array_input_takes_a_whole_vector_per_row() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"codes": {"data_type": "int_array"}}}"#),
            &[spec("codes", ElementType::I32, Some(2))],
        )
        .expect("two codes per row");

        let batch = builder
            .build(&request(&[row("r1", &[("codes", V::I32(vec![3, 4]))])]))
            .expect("one row");
        assert_eq!(
            batch.get("codes").expect("codes").data,
            FeatureData::I32(vec![3, 4])
        );
    }

    /// A ragged column -- rows disagreeing with each other -- cannot be sent at all now,
    /// so what is left to catch is a column whose width disagrees with the model's.
    #[test]
    fn a_text_column_of_the_wrong_width_is_refused_naming_the_counts() {
        let builder = InputBuilder::resolve(
            config(r#"{"model_inputs": {"tags": {"data_type": "string_array"}}}"#),
            &[spec("tags", ElementType::Str, Some(2))],
        )
        .expect("two strings per row");

        let err = builder
            .build(&test_support::columns(
                &["r1", "r2"],
                vec![("tags", 1, V::Str(vec!["a".into(), "c".into()]))],
            ))
            .expect_err("one string per row is not two");
        let m = err.to_string();
        assert!(
            m.contains("tags") && m.contains('4') && m.contains('2'),
            "the error should say how many values it wanted and how many arrived: {m}"
        );
    }

    #[test]
    fn a_dynamic_width_with_nothing_to_size_it_is_refused_at_startup() {
        let err = InputBuilder::resolve(
            config(r#"{"model_inputs": {"tags": {"data_type": "string_array"}}}"#),
            &[spec("tags", ElementType::Str, None)],
        )
        .expect_err("an absent feature would have no shape to fill");
        let m = err.to_string();
        assert!(m.contains("tags") && m.contains("dynamic"), "got {m}");
    }

    /// A named input whose model width is dynamic is refused at startup, not on the first
    /// request.
    #[test]
    fn a_dynamic_width_is_refused_for_a_named_input() {
        let err = InputBuilder::resolve(
            config(r#"{"model_inputs": {"city": {"data_type": "float_array"}}}"#),
            &[spec("city", ElementType::F32, None)],
        )
        .expect_err("a dynamic width has no shape to fill verbatim");
        let m = err.to_string();
        assert!(
            m.contains("city") && m.contains("dynamic"),
            "the refusal should name the input and its dynamic width; got {m}"
        );
    }

    /// `feature_order` transforms; `model_inputs` passes values through and is refused
    /// outright if it carries a transformation.
    #[test]
    fn only_a_concatenated_input_is_transformed() {
        let scaled = InputBuilder::resolve(
            config(
                r#"{
                    "feature_transformations": {
                        "age": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.0]}
                    },
                    "feature_order": ["age"]
                }"#,
            ),
            &[spec("features", ElementType::F32, Some(1))],
        )
        .expect("feature_order transforms")
        .build(&request(&[row("r1", &[("age", V::f32(50.0))])]))
        .expect("one row");
        assert_eq!(
            scaled.get("features").expect("features").data,
            FeatureData::F32(vec![0.5]),
            "feature_order is the one shape that transforms"
        );

        let passed = InputBuilder::resolve(
            config(r#"{"model_inputs": {"age": {"data_type": "float"}}}"#),
            &[spec("age", ElementType::F32, Some(1))],
        )
        .expect("model_inputs resolves")
        .build(&request(&[row("r1", &[("age", V::f32(50.0))])]))
        .expect("one row");
        assert_eq!(
            passed.get("age").expect("age").data,
            FeatureData::F32(vec![50.0]),
            "model_inputs must pass the value through untouched"
        );

        VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {"age": {"type": "identity", "default_val": [0.0]}},
                "model_inputs": {"age": {"data_type": "float"}}
            }"#,
        )
        .expect_err("model_inputs cannot carry transformations");
    }

    #[test]
    fn a_declared_input_takes_its_like_named_feature() {
        let builder = InputBuilder::resolve(
            config(
                r#"{"model_inputs": {
                    "numbers": {"data_type": "float_array"},
                    "tags": {"data_type": "string"}
                }}"#,
            ),
            &[
                spec("numbers", ElementType::F32, Some(2)),
                spec("tags", ElementType::Str, Some(1)),
            ],
        )
        .expect("a signature is enough");

        let batch = builder
            .build(&request(&[row(
                "r1",
                &[
                    ("numbers", V::F32(vec![1.5, 2.5])),
                    ("tags", V::str("beta")),
                ],
            )]))
            .expect("one row");

        assert_eq!(
            batch.get("numbers").expect("numbers").data,
            FeatureData::F32(vec![1.5, 2.5])
        );
        assert_eq!(
            batch.get("tags").expect("tags").data,
            FeatureData::Str(vec!["beta".into()])
        );
    }

    #[test]
    fn a_model_with_no_inputs_is_refused() {
        let err = InputBuilder::resolve(config(r#"{"feature_order": ["a"]}"#), &[])
            .expect_err("nothing to feed");
        assert!(err.to_string().contains("no inputs"), "got {err}");
    }

    #[test]
    fn a_near_miss_is_suggested_and_a_stranger_is_not() {
        let declared = ["oh_color", "sd32_age", "text_title"];
        assert_eq!(
            did_you_mean("oh_colour", declared.into_iter()),
            " Did you mean \"oh_color\"?",
            "a one-letter typo is the whole point of the check"
        );
        assert_eq!(
            did_you_mean("sd32_agee", declared.into_iter()),
            " Did you mean \"sd32_age\"?"
        );
        assert_eq!(
            did_you_mean("completely_different", declared.into_iter()),
            "",
            "a genuinely absent input should suggest nothing rather than mislead"
        );
    }

    #[test]
    fn the_banner_says_where_each_input_comes_from() {
        let builder = InputBuilder::resolve(
            config(
                r#"{
                    "model_inputs": {
                        "age": {"data_type": "float"},
                        "tags": {"data_type": "string"}
                    }
                }"#,
            ),
            &[
                spec("age", ElementType::F32, Some(1)),
                spec("tags", ElementType::Str, Some(1)),
            ],
        )
        .expect("both declared");

        let described = builder.describe();
        assert!(described.contains("age FP32[1]"), "got {described}");
        assert_eq!(
            described.matches("verbatim").count(),
            2,
            "both named inputs are verbatim now, the float one included; got {described}"
        );
        assert!(
            !described.contains("transformed"),
            "model_inputs never transforms; got {described}"
        );
    }
}
