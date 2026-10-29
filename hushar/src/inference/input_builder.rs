// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Turning a request's features into the model's input.
//!
//! Everything that does not depend on a request is settled once, at startup, by
//! [`InputBuilder::resolve`]: which features feed the model's input, its element type,
//! and how many values per row it carries. A request then only walks that list.
//!
//! The configuration declares the mapping; it is checked against the model's signature,
//! never inferred from it.
//!
//! > Every feature named in `feature_order` is transformed and concatenated into the
//! > model's one input, so a row reaches the engine as a numeric vector and a feature a
//! > request omits contributes its transformation's default.

use std::collections::HashMap;
use std::ops::Range;

use crate::config::vectorization_config::{VectorType, VectorizationConfig};
use crate::inference::InferenceError;
use crate::inference::batch::{FeatureData, InputBatch, IoSpec};
use crate::inference::proto_columns;
use crate::inference::transformations::Transformation;
use hushar::hushar_proto::Feature;

/// Builds the model's input from a batch of feature rows.
///
/// # Fields
///
/// * `name` - the model's name for its one input
/// * `data_type` - the float type of the concatenated vector, which chooses the buffer
///   to fill
/// * `width` - values per row, the sum of the configured features' widths
/// * `plan` - every feature paired with the transformation that produces its values, in
///   the order they are concatenated
#[derive(Debug)]
pub(crate) struct InputBuilder {
    name: String,
    data_type: VectorType,
    width: usize,
    plan: Vec<(String, Box<dyn Transformation>)>,
}

impl InputBuilder {
    /// Checks a configuration against a model's signature and settles the mapping.
    ///
    /// The width is the sum of the configured features' widths, and must equal the width
    /// the model declares. Each feature is paired with its transformation here, so a
    /// request resolves none.
    ///
    /// # Errors
    ///
    /// Fails unless the model declares exactly one input, if that input's element type
    /// is not the configured `data_type`, or if the configured features produce no values
    /// or a width the model does not take.
    pub(crate) fn resolve(
        config: VectorizationConfig,
        specs: &[IoSpec],
    ) -> Result<Self, InferenceError> {
        let [spec] = specs else {
            return Err(not_one_input(specs));
        };

        let element = config.data_type.element_type();
        if spec.element_type != element {
            return Err(format!(
                "\"data_type\": {:?} makes the batch {element}, but model input {:?} is {}",
                config.data_type.name(),
                spec.name,
                spec.element_type
            )
            .into());
        }

        let width = config.vector_width()?;
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

        let VectorizationConfig {
            feature_order,
            data_type,
            mut transformations,
        } = config;
        let mut plan = Vec::with_capacity(feature_order.len());
        for feature in feature_order {
            let transformation = transformations.remove(&feature).ok_or_else(|| {
                format!("feature {feature:?} in feature_order has no transformation")
            })?;
            plan.push((feature, transformation));
        }

        Ok(Self {
            name: spec.name.clone(),
            data_type,
            width,
            plan,
        })
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

    /// Builds the model's input from the rows in `chunk`, `pad_to` rows tall.
    ///
    /// Nothing is hashed here: the columns were indexed once when the request arrived.
    ///
    /// `pad_to` is at least the chunk's height, and any excess is **ballast**: a pinned
    /// model wants a fixed batch height, so the rows past the real ones are scored and
    /// thrown away without ever reaching a response or a log. Ballast is written at the
    /// end of the buffer -- one row of every transformation's default, repeated, which is
    /// exactly what a row sending no features would have produced.
    ///
    /// # Errors
    ///
    /// Fails if a column does not carry a whole number of rows, if a transformation
    /// refuses a value, or if a row does not produce the width settled at startup.
    pub(crate) fn build_padded(
        &self,
        columns: &RequestColumns,
        chunk: Range<usize>,
        pad_to: usize,
    ) -> Result<InputBatch, InferenceError> {
        let height = chunk.end.saturating_sub(chunk.start);
        debug_assert!(pad_to >= height, "padding cannot drop rows");
        let pad_to = pad_to.max(height);

        let bound = self.bind(columns)?;
        let data = match self.data_type {
            VectorType::Float => FeatureData::F32(self.values_f32(&bound, columns, chunk, pad_to)?),
            VectorType::Double => {
                FeatureData::F64(self.values_f64(&bound, columns, chunk, pad_to)?)
            }
        };

        let mut batch = InputBatch::new(pad_to);
        batch.push(self.name.clone(), data)?;
        Ok(batch)
    }

    /// How the model's input is fed, for the startup banner.
    ///
    /// The features are named in full up to eight of them, and summarised beyond that.
    pub(crate) fn describe(&self) -> String {
        const NAMED: usize = 8;
        let names: Vec<&str> = self.plan.iter().map(|(name, _)| name.as_str()).collect();
        let how = if names.len() <= NAMED {
            format!("[{}] transformed and concatenated", names.join(", "))
        } else {
            format!(
                "{} features transformed and concatenated [{}, ... +{} more]",
                names.len(),
                names[..NAMED].join(", "),
                names.len() - NAMED,
            )
        };
        format!(
            "{} {}[{}] <- {how}",
            self.name,
            self.data_type.element_type(),
            self.width
        )
    }

    /// Pairs every planned feature with its column, and settles each one's width.
    ///
    /// # Errors
    ///
    /// Fails if a column's length is not a whole number of rows.
    fn bind<'a>(&'a self, columns: &'a RequestColumns) -> Result<Bound<'a>, InferenceError> {
        let rows = columns.rows();
        let mut bound = Vec::with_capacity(self.plan.len());
        for (name, transformation) in &self.plan {
            let found = match columns.get(name) {
                None => None,
                Some(data) if rows == 0 => Some((data, 0)),
                Some(data) => {
                    if !data.len().is_multiple_of(rows) {
                        return Err(format!(
                            "feature {name:?} carries {} values, which is not a whole \
                             number of rows for a request of {rows}",
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

    /// The concatenated input: every feature transformed, in order.
    ///
    /// Each transformation appends straight into the buffer being built, and an absent
    /// feature contributes its default.
    ///
    /// # Errors
    ///
    /// Fails if a column runs short of the row asked for, if a transformation refuses a
    /// value, or if a row does not produce the input's width.
    fn values_f32(
        &self,
        bound: &Bound<'_>,
        columns: &RequestColumns,
        chunk: Range<usize>,
        pad_to: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let mut values = Vec::with_capacity(pad_to * self.width);
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
                                format!("feature {name:?} in row {:?}: {e}", columns.row_id(row))
                                    .into()
                            },
                        )?;
                    }
                    None => values.extend_from_slice(transformation.get_default_val()),
                }
            }
            self.check_row(columns.row_id(row), values.len() - before)?;
        }
        pad_with(&mut values, pad_to * self.width, &self.default_row_f32());
        Ok(values)
    }

    /// The same, in double precision.
    fn values_f64(
        &self,
        bound: &Bound<'_>,
        columns: &RequestColumns,
        chunk: Range<usize>,
        pad_to: usize,
    ) -> Result<Vec<f64>, InferenceError> {
        let mut values = Vec::with_capacity(pad_to * self.width);
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
                                format!("feature {name:?} in row {:?}: {e}", columns.row_id(row))
                                    .into()
                            },
                        )?;
                    }
                    None => values.extend(transformation.default_val_f64()),
                }
            }
            self.check_row(columns.row_id(row), values.len() - before)?;
        }
        pad_with(&mut values, pad_to * self.width, &self.default_row_f64());
        Ok(values)
    }

    /// One ballast row: every transformation's default, concatenated.
    fn default_row_f32(&self) -> Vec<f32> {
        self.plan
            .iter()
            .flat_map(|(_, t)| t.get_default_val().iter().copied())
            .collect()
    }

    /// The same, in double precision.
    fn default_row_f64(&self) -> Vec<f64> {
        self.plan
            .iter()
            .flat_map(|(_, t)| t.default_val_f64())
            .collect()
    }

    /// Checks that a row produced the width settled at startup.
    ///
    /// # Errors
    ///
    /// Fails when it did not: a batch whose rows differ in width has no rectangular
    /// shape.
    fn check_row(&self, row_id: &str, produced: usize) -> Result<(), InferenceError> {
        if produced == self.width {
            return Ok(());
        }
        Err(format!(
            "model input {:?} takes {} values per row, but row {row_id:?} produced {produced}",
            self.name, self.width
        )
        .into())
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

/// A signature the configuration cannot feed, because it is not one input.
///
/// Features are concatenated into a single vector, so a model that takes several inputs
/// -- or none -- has no shape this service can fill.
fn not_one_input(specs: &[IoSpec]) -> InferenceError {
    if specs.is_empty() {
        return "the model declares no inputs, so there is nothing to feed".into();
    }
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    format!(
        "every feature is transformed and concatenated into one input, but the model \
         declares {}: {}. Export the model with a single input that takes the whole \
         feature vector.",
        specs.len(),
        names.join(", ")
    )
    .into()
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
    /// column is checked to hold a whole number of rows.
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
    use crate::inference::batch::ElementType;

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

    /// A model taking several inputs has no shape one concatenated vector can fill.
    #[test]
    fn a_multi_input_model_is_refused_naming_its_inputs() {
        let err = InputBuilder::resolve(
            config(r#"{"feature_order": ["a"]}"#),
            &[
                spec("numbers", ElementType::F32, Some(1)),
                spec("tags", ElementType::F32, Some(1)),
            ],
        )
        .expect_err("one vector cannot feed two inputs");
        let m = err.to_string();
        assert!(
            m.contains("numbers") && m.contains("tags"),
            "the error must name the inputs it found: {m}"
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

    /// A model whose width is dynamic takes the width the configuration settles.
    #[test]
    fn a_dynamic_model_width_is_settled_by_the_configuration() {
        let batch = InputBuilder::resolve(
            config(r#"{"feature_order": ["a", "b"]}"#),
            &[spec("features", ElementType::F32, None)],
        )
        .expect("the configuration supplies the width")
        .build(&request(&[row(
            "r1",
            &[("a", V::f32(1.0)), ("b", V::f32(2.0))],
        )]))
        .expect("one row");
        assert_eq!(batch.get("features").expect("features").width(), 2);
    }

    /// Ballast is each transformation's default, which is what a featureless row gives.
    #[test]
    fn ballast_repeats_what_a_featureless_row_would_have_produced() {
        let batch = InputBuilder::resolve(
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
            batch.get("features").expect("features").data,
            FeatureData::F32(vec![1.0, 7.0, 7.0]),
            "ballast is the transformation's default, not a zero"
        );
        assert_eq!(batch.rows(), 3, "the engine is given the padded height");
    }

    /// A 3-wide input pads in whole rows of three, so no later row is read shifted.
    #[test]
    fn ballast_lands_on_a_row_boundary_for_a_wide_input() {
        let batch = InputBuilder::resolve(
            config(
                r#"{
                    "feature_transformations": {
                        "city": {"type": "one_hot_encoding", "categories": ["a", "b", "c"], "default_val": [0, 0, 0]}
                    },
                    "feature_order": ["city"]
                }"#,
            ),
            &[spec("features", ElementType::F32, Some(3))],
        )
        .expect("three values per row")
        .build_padded(&request(&[row("r1", &[("city", V::str("b"))])]), 0..1, 3)
        .expect("one real row, two of ballast");
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F32(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            batch.get("features").expect("features").width(),
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

    /// A row wider than the settled width is refused with both numbers.
    ///
    /// The width comes from the configuration, so a request sending more values for a
    /// feature than its transformation was sized for produces a row that cannot be
    /// stacked with the others.
    #[test]
    fn a_row_that_produces_the_wrong_width_is_refused() {
        let builder = InputBuilder::resolve(
            config(r#"{"feature_order": ["v"]}"#),
            &[spec("features", ElementType::F32, Some(1))],
        )
        .expect("one value per row");

        let err = builder
            .build(&test_support::columns(
                &["r1", "r2"],
                vec![("v", 2, V::F32(vec![1.0, 2.0, 3.0, 4.0]))],
            ))
            .expect_err("2 values per row is not the 1 the input takes");
        let m = err.to_string();
        assert!(
            m.contains("r1") && m.contains("produced 2"),
            "the error should name the row and both widths: {m}"
        );
    }

    /// `row_ids` states the row count, and every feature is measured against it.
    #[test]
    fn row_ids_state_the_row_count() {
        let builder = InputBuilder::resolve(
            config(r#"{"feature_order": ["v"]}"#),
            &[spec("features", ElementType::F32, Some(1))],
        )
        .expect("one value per row");

        let columns = test_support::columns(
            &["r1", "r2", "r3"],
            vec![("v", 1, V::F32(vec![1.0, 2.0, 3.0]))],
        );
        assert_eq!(columns.rows(), 3, "three ids means three rows");

        let batch = builder.build(&columns).expect("3 values is 3 rows of 1");
        assert_eq!(batch.rows(), 3);
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F32(vec![1.0, 2.0, 3.0])
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

    /// Every feature reaches the model through its transformation, never as sent.
    #[test]
    fn the_input_carries_transformed_values() {
        let batch = InputBuilder::resolve(
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
        .expect("one concatenated input")
        .build(&request(&[row("r1", &[("age", V::f32(50.0))])]))
        .expect("one row");
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F32(vec![0.5]),
            "the value is scaled on its way in"
        );
    }

    /// An integer feature reaches the vector through its transformation, as a float.
    #[test]
    fn integer_features_reach_the_vector_as_floats() {
        let batch = InputBuilder::resolve(
            config(r#"{"feature_order": ["n", "code"]}"#),
            &[spec("features", ElementType::F32, Some(2))],
        )
        .expect("two features into one 2-wide input")
        .build(&request(&[row(
            "r1",
            &[("n", V::i32(7)), ("code", V::i64(9))],
        )]))
        .expect("one row");
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F32(vec![7.0, 9.0])
        );
    }

    #[test]
    fn a_model_with_no_inputs_is_refused() {
        let err = InputBuilder::resolve(config(r#"{"feature_order": ["a"]}"#), &[])
            .expect_err("nothing to feed");
        assert!(err.to_string().contains("no inputs"), "got {err}");
    }

    #[test]
    fn the_banner_names_the_features_the_input_carries() {
        let described = InputBuilder::resolve(
            config(r#"{"feature_order": ["age", "city"]}"#),
            &[spec("features", ElementType::F32, Some(2))],
        )
        .expect("two features into one input")
        .describe();

        assert_eq!(
            described,
            "features FP32[2] <- [age, city] transformed and concatenated"
        );
    }

    /// Past eight features the banner gives a count rather than the whole list.
    #[test]
    fn the_banner_summarises_a_wide_input() {
        let order: Vec<String> = (0..10).map(|i| format!("\"f{i}\"")).collect();
        let described = InputBuilder::resolve(
            config(&format!(r#"{{"feature_order": [{}]}}"#, order.join(", "))),
            &[spec("features", ElementType::F32, Some(10))],
        )
        .expect("ten features into one input")
        .describe();

        assert!(
            described.contains("10 features transformed and concatenated")
                && described.contains("f0, f1, f2, f3, f4, f5, f6, f7, ... +2 more"),
            "got {described}"
        );
    }
}
