// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Turning a request's features into the model's inputs.
//!
//! Everything that does not depend on a request is settled once, at startup, by
//! [`InputBuilder::resolve`]: which feature feeds which input, what each input's
//! element type is, and how many values per row it carries. A request then only
//! walks that list.
//!
//! Nothing here is inferred. The configuration declares the mapping and this checks it
//! against the model's signature. An earlier version read the signature and guessed,
//! which needed an error for every way the guess could be ambiguous; a declaration
//! cannot be ambiguous, so those errors are gone rather than reworded.
//!
//! The one rule worth stating on its own:
//!
//! > A float input is built by a transformation. Any other input is passed through
//! > from the request verbatim.
//!
//! Transformations are defined over floats, so rounding one into an integer or
//! inventing text from one would be silent corruption. That is not a third kind of
//! binding, it follows from the input's element type, so it is written once in
//! [`InputBuilder::build`] rather than encoded in the plan.

use hushar::hushar_proto::{DataType as WireValue, InputRow, data_type::DataType as Value};
use std::collections::HashMap;

use crate::config::vectorization_config::{
    InputDataType, InputMode, VectorizationConfig, identity_of_width,
};
use crate::inference::InferenceError;
use crate::inference::batch::{ElementType, FeatureData, InputBatch, IoSpec};
use crate::inference::transformations::Transformation;

/// Where one model input's values come from.
///
/// Variants:
/// - `Vectorised` — every feature named, transformed and concatenated in that order.
/// - `Feature` — the one feature of that name.
#[derive(Clone, Debug, PartialEq)]
enum Source {
    Vectorised(Vec<String>),
    Feature(String),
}

/// One model input, fully settled.
///
/// Fields:
/// - `name` — the model's name for this input.
/// - `element` — its element type, which also decides transform or verbatim.
/// - `width` — values per row, settled at startup from the model or from the
///   configuration. Knowing it before any request arrives is what lets a row be checked
///   against a definite number rather than against whatever the first row produced.
/// - `source` — where the values come from.
#[derive(Debug)]
struct PlannedInput {
    name: String,
    element: ElementType,
    width: usize,
    source: Source,
}

/// Builds a batch of model inputs from a batch of feature rows.
///
/// Owns the transformations, including the identity ones filled in for features
/// that declared none, so building needs nothing but a request.
#[derive(Debug)]
pub(crate) struct InputBuilder {
    inputs: Vec<PlannedInput>,
    transformations: HashMap<String, Box<dyn Transformation>>,
}

impl InputBuilder {
    /// Checks a configuration against a model's signature and settles the mapping.
    ///
    /// `config` is `None` for a deployment with no vectorization configuration, which
    /// feeds every model input from the feature of the same name, verbatim.
    pub(crate) fn resolve(
        config: Option<VectorizationConfig>,
        specs: &[IoSpec],
    ) -> Result<Self, InferenceError> {
        if specs.is_empty() {
            return Err("the model declares no inputs, so there is nothing to feed".into());
        }

        match config {
            None => Self::pass_everything_through(specs),
            Some(config) => {
                let VectorizationConfig {
                    feature_transformations,
                    mode,
                } = config;
                match mode {
                    InputMode::Vectorised {
                        feature_order,
                        data_type,
                    } => Self::vectorised(feature_transformations, feature_order, data_type, specs),
                    InputMode::Named { model_inputs } => {
                        Self::named(feature_transformations, model_inputs, specs)
                    }
                }
            }
        }
    }

    /// Every model input fed by the feature of the same name.
    ///
    /// What `vectorization_config` being absent means. A float input still needs a
    /// transformation, so it gets an identity one sized from the model -- the value
    /// the caller sent when the feature is present, zeros of the right width when it
    /// is not.
    fn pass_everything_through(specs: &[IoSpec]) -> Result<Self, InferenceError> {
        let mut transformations: HashMap<String, Box<dyn Transformation>> = HashMap::new();
        let mut inputs = Vec::with_capacity(specs.len());

        for spec in specs {
            let width = required_width(spec, None)?;
            if spec.element_type.is_float() {
                transformations.insert(spec.name.clone(), identity_of_width(width));
            }
            inputs.push(PlannedInput {
                name: spec.name.clone(),
                element: spec.element_type,
                width,
                source: Source::Feature(spec.name.clone()),
            });
        }

        Ok(Self {
            inputs,
            transformations,
        })
    }

    /// One input carrying every feature concatenated.
    ///
    /// The width is summed by the same function the configuration uses, so the two
    /// cannot disagree about the model's input width.
    fn vectorised(
        transformations: HashMap<String, Box<dyn Transformation>>,
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

        Ok(Self {
            inputs: vec![PlannedInput {
                name: spec.name.clone(),
                element,
                width,
                source: Source::Vectorised(feature_order),
            }],
            transformations,
        })
    }

    /// One input per named feature.
    ///
    /// Both directions of the declaration are checked, because the two mistakes differ:
    /// an undeclared input would be left unfed, and a declared input the model does not
    /// have is usually a typo that would otherwise pass unnoticed.
    ///
    /// A float input with no transformation of its own gets an identity one, sized from
    /// the model so an absent feature still fills the row.
    fn named(
        mut transformations: HashMap<String, Box<dyn Transformation>>,
        model_inputs: std::collections::BTreeMap<String, InputDataType>,
        specs: &[IoSpec],
    ) -> Result<Self, InferenceError> {
        for spec in specs {
            if !model_inputs.contains_key(&spec.name) {
                let declared: Vec<&str> = model_inputs.keys().map(String::as_str).collect();
                return Err(format!(
                    "model input {:?} is not declared in model_inputs, so nothing would \
                     feed it. Declared: {}",
                    spec.name,
                    declared.join(", ")
                )
                .into());
            }
        }
        for name in model_inputs.keys() {
            if !specs.iter().any(|s| &s.name == name) {
                let actual: Vec<String> = specs.iter().map(IoSpec::describe).collect();
                return Err(format!(
                    "model_inputs declares {name:?}, but the model has no such input. \
                     The model takes: {}",
                    actual.join(", ")
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

            let configured = transformations
                .get(&spec.name)
                .map(|t| t.get_default_val().len());
            let width = required_width(spec, configured)?;

            if !declared.is_array() && width != 1 {
                return Err(format!(
                    "model input {:?} is declared {declared}, which is one value per \
                     row, but {width} values per row are produced for it. Declare it as \
                     {}_array, or narrow the transformation.",
                    spec.name,
                    declared.name()
                )
                .into());
            }
            if let (Some(declared_width), Some(configured_width)) = (spec.width, configured)
                && declared_width != configured_width
            {
                return Err(format!(
                    "model input {:?} takes {declared_width} values per row, but its \
                     transformation produces {configured_width}",
                    spec.name
                )
                .into());
            }

            if spec.element_type.is_float() {
                transformations
                    .entry(spec.name.clone())
                    .or_insert_with(|| identity_of_width(width));
            }

            inputs.push(PlannedInput {
                name: spec.name.clone(),
                element: spec.element_type,
                width,
                source: Source::Feature(spec.name.clone()),
            });
        }

        Ok(Self {
            inputs,
            transformations,
        })
    }

    /// Builds one batch of model inputs.
    ///
    /// The transform-or-verbatim rule lives here, in one `match` on the element type,
    /// rather than being decided again per input.
    pub(crate) fn build(&self, rows: &[InputRow]) -> Result<InputBatch, InferenceError> {
        let mut batch = InputBatch::new(rows.len());

        for input in &self.inputs {
            let data = match input.element {
                ElementType::F32 => FeatureData::F32(self.build_f32(input, rows)?),
                ElementType::F64 => FeatureData::F64(self.build_f64(input, rows)?),
                ElementType::I32 => FeatureData::I32(build_i32(input, rows)?),
                ElementType::I64 => FeatureData::I64(build_i64(input, rows)?),
                ElementType::Str => FeatureData::Str(build_str(input, rows)?),
            };
            batch.push(input.name.clone(), data)?;
        }
        Ok(batch)
    }

    /// How each model input is fed, for the startup banner.
    pub(crate) fn describe(&self) -> String {
        self.inputs
            .iter()
            .map(|input| {
                let how = match &input.source {
                    Source::Vectorised(order) => {
                        // Named in full while the list is short enough to read, and
                        // summarised past that. A 200-feature deployment otherwise puts
                        // every name on one line thousands of characters wide, which is
                        // not a contract anyone checks -- and the order is in the
                        // configuration, which is the authority anyway.
                        const NAMED: usize = 8;
                        if order.len() <= NAMED {
                            format!("[{}] transformed and concatenated", order.join(", "))
                        } else {
                            format!(
                                "{} features transformed and concatenated [{}, ... +{} more]",
                                order.len(),
                                order[..NAMED].join(", "),
                                order.len() - NAMED,
                            )
                        }
                    }
                    Source::Feature(name) if input.element.is_float() => {
                        format!("feature {name:?}, transformed")
                    }
                    Source::Feature(name) => format!("feature {name:?}, verbatim"),
                };
                format!("{} {}[{}] <- {how}", input.name, input.element, input.width)
            })
            .collect::<Vec<_>>()
            .join("\n             ")
    }

    /// The transformation for a feature, which resolve guaranteed is present.
    fn transformation(&self, feature: &str) -> Result<&dyn Transformation, InferenceError> {
        self.transformations
            .get(feature)
            .map(AsRef::as_ref)
            .ok_or_else(|| {
                format!(
                    "feature {feature:?} has no transformation, which startup should \
                     have caught"
                )
                .into()
            })
    }

    fn build_f32(
        &self,
        input: &PlannedInput,
        rows: &[InputRow],
    ) -> Result<Vec<f32>, InferenceError> {
        let mut values = Vec::with_capacity(rows.len() * input.width);
        for row in rows {
            let before = values.len();
            match &input.source {
                Source::Vectorised(order) => {
                    for feature in order {
                        let transformation = self.transformation(feature)?;
                        values.extend(apply(transformation, feature, &row.features)?);
                    }
                }
                Source::Feature(feature) => {
                    let transformation = self.transformation(feature)?;
                    values.extend(apply(transformation, feature, &row.features)?);
                }
            }
            check_row(input, row, values.len() - before)?;
        }
        Ok(values)
    }

    fn build_f64(
        &self,
        input: &PlannedInput,
        rows: &[InputRow],
    ) -> Result<Vec<f64>, InferenceError> {
        let mut values = Vec::with_capacity(rows.len() * input.width);
        for row in rows {
            let before = values.len();
            match &input.source {
                Source::Vectorised(order) => {
                    for feature in order {
                        let transformation = self.transformation(feature)?;
                        values.extend(apply_f64(transformation, feature, &row.features)?);
                    }
                }
                Source::Feature(feature) => {
                    let transformation = self.transformation(feature)?;
                    values.extend(apply_f64(transformation, feature, &row.features)?);
                }
            }
            check_row(input, row, values.len() - before)?;
        }
        Ok(values)
    }
}

/// The single feature feeding a verbatim input.
///
/// Only a float batch can be concatenated, which `vectorised` enforces, so the other
/// arm is defensive rather than reachable.
fn verbatim_feature(input: &PlannedInput) -> Result<&str, InferenceError> {
    match &input.source {
        Source::Feature(name) => Ok(name),
        Source::Vectorised(_) => Err(format!(
            "model input {:?} is {} and cannot be built by concatenating \
             transformations, which produce floats",
            input.name, input.element
        )
        .into()),
    }
}

/// A 32-bit integer input, read from the request verbatim.
///
/// An `integer` feature only. A `long` is **not** accepted: narrowing it could
/// overflow, and a category code that silently wrapped would be a wrong prediction
/// rather than an error. Widening the other way is exact, which is why [`build_i64`]
/// does accept an integer.
fn build_i32(input: &PlannedInput, rows: &[InputRow]) -> Result<Vec<i32>, InferenceError> {
    let feature = verbatim_feature(input)?;
    let mut values = Vec::with_capacity(rows.len() * input.width);
    for row in rows {
        let before = values.len();
        match row.features.get(feature).and_then(|f| f.data_type.as_ref()) {
            None => values.extend(std::iter::repeat_n(0i32, input.width)),
            Some(Value::IntegerValue(v)) => values.push(*v),
            Some(Value::IntegerArray(a)) => values.extend(a.values.iter().copied()),
            Some(other) => {
                return Err(format!(
                    "model input {:?} is INT32 and is fed verbatim, so feature \
                     {feature:?} must be an integer, but row {:?} sent {}",
                    input.name,
                    row.row_id,
                    describe_value(other)
                )
                .into());
            }
        }
        check_row(input, row, values.len() - before)?;
    }
    Ok(values)
}

/// A 64-bit integer input, read from the request verbatim.
///
/// An `integer` feature is accepted as well as a `long`, because widening it is exact.
fn build_i64(input: &PlannedInput, rows: &[InputRow]) -> Result<Vec<i64>, InferenceError> {
    let feature = verbatim_feature(input)?;
    let mut values = Vec::with_capacity(rows.len() * input.width);
    for row in rows {
        let before = values.len();
        match row.features.get(feature).and_then(|f| f.data_type.as_ref()) {
            None => values.extend(std::iter::repeat_n(0i64, input.width)),
            Some(Value::LongValue(v)) => values.push(*v),
            Some(Value::IntegerValue(v)) => values.push(i64::from(*v)),
            Some(Value::LongArray(a)) => values.extend(a.values.iter().copied()),
            Some(Value::IntegerArray(a)) => {
                values.extend(a.values.iter().copied().map(i64::from));
            }
            Some(other) => {
                return Err(format!(
                    "model input {:?} is INT64 and is fed verbatim, so feature \
                     {feature:?} must be a long or an integer, but row {:?} sent {}",
                    input.name,
                    row.row_id,
                    describe_value(other)
                )
                .into());
            }
        }
        check_row(input, row, values.len() - before)?;
    }
    Ok(values)
}

/// A text input, read from the request verbatim.
///
/// An absent feature becomes the empty string rather than a refusal: a partial request
/// is normal on a serving path and a refused one is not.
fn build_str(input: &PlannedInput, rows: &[InputRow]) -> Result<Vec<String>, InferenceError> {
    let feature = verbatim_feature(input)?;
    let mut values = Vec::with_capacity(rows.len() * input.width);
    for row in rows {
        let before = values.len();
        match row.features.get(feature).and_then(|f| f.data_type.as_ref()) {
            None => values.extend(std::iter::repeat_n(String::new(), input.width)),
            Some(Value::StringValue(s)) => values.push(s.clone()),
            Some(Value::StringArray(a)) => values.extend(a.values.iter().cloned()),
            Some(other) => {
                return Err(format!(
                    "model input {:?} is STRING and is fed verbatim, so feature \
                     {feature:?} must be a string, but row {:?} sent {}",
                    input.name,
                    row.row_id,
                    describe_value(other)
                )
                .into());
            }
        }
        check_row(input, row, values.len() - before)?;
    }
    Ok(values)
}

/// One feature's transformed values, or its default when it is absent.
///
/// The default covers both an absent feature and a present-but-unset one, so a
/// partial request scores rather than failing.
fn apply(
    transformation: &dyn Transformation,
    feature: &str,
    features: &HashMap<String, WireValue>,
) -> Result<Vec<f32>, InferenceError> {
    match features.get(feature).and_then(|f| f.data_type.as_ref()) {
        Some(value) => transformation
            .transform(value)
            .map_err(|e| -> InferenceError { format!("feature {feature:?}: {e}").into() }),
        None => Ok(transformation.get_default_val().to_vec()),
    }
}

/// The same, in double precision.
fn apply_f64(
    transformation: &dyn Transformation,
    feature: &str,
    features: &HashMap<String, WireValue>,
) -> Result<Vec<f64>, InferenceError> {
    match features.get(feature).and_then(|f| f.data_type.as_ref()) {
        Some(value) => transformation
            .transform_f64(value)
            .map_err(|e| -> InferenceError { format!("feature {feature:?}: {e}").into() }),
        None => Ok(transformation.default_val_f64()),
    }
}

/// Every row must be the width settled at startup.
///
/// A batch whose rows differ in width has no rectangular shape, so the alternative
/// to this error is a buffer that the engine reads as though it were rectangular.
fn check_row(input: &PlannedInput, row: &InputRow, produced: usize) -> Result<(), InferenceError> {
    if produced == input.width {
        return Ok(());
    }
    Err(format!(
        "model input {:?} takes {} values per row, but row {:?} produced {produced}",
        input.name, input.width, row.row_id
    )
    .into())
}

/// Values per row for one input, from the model or from configuration.
///
/// The model is preferred where it fixes the axis, because that is the number the
/// engine will actually read. Where the model leaves the axis dynamic, the
/// configuration decides -- and where neither does, this fails at startup rather
/// than letting the first absent feature produce a ragged batch.
fn required_width(spec: &IoSpec, configured: Option<usize>) -> Result<usize, InferenceError> {
    match (spec.width, configured) {
        (Some(w), _) => Ok(w),
        (None, Some(w)) => Ok(w),
        (None, None) => Err(format!(
            "model input {:?} leaves its width dynamic and nothing configures one, so \
             an absent feature would have no shape to fill. Export the model with a \
             fixed width for this input, or give the feature a transformation whose \
             default_val is as wide as the input.",
            spec.name
        )
        .into()),
    }
}

/// The configuration spelling of a vector's element type, for an error.
fn data_type_name(data_type: crate::config::vectorization_config::VectorType) -> &'static str {
    match data_type {
        crate::config::vectorization_config::VectorType::Float => "float",
        crate::config::vectorization_config::VectorType::Double => "double",
    }
}

/// Names a feature value's variant, for an error message.
fn describe_value(value: &Value) -> &'static str {
    match value {
        Value::DoubleValue(_) => "a double",
        Value::FloatValue(_) => "a float",
        Value::IntegerValue(_) => "an integer",
        Value::LongValue(_) => "a long",
        Value::StringValue(_) => "a string",
        Value::BoolValue(_) => "a bool",
        Value::DoubleArray(_) => "a double array",
        Value::FloatArray(_) => "a float array",
        Value::IntegerArray(_) => "an integer array",
        Value::LongArray(_) => "a long array",
        Value::StringArray(_) => "a string array",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hushar::hushar_proto::{FloatArray, StringArray};

    fn spec(name: &str, element_type: ElementType, width: Option<usize>) -> IoSpec {
        IoSpec::new(name, element_type, width)
    }

    fn config(json: &str) -> VectorizationConfig {
        VectorizationConfig::from_json(json).expect("valid configuration")
    }

    fn row(row_id: &str, features: &[(&str, Value)]) -> InputRow {
        InputRow {
            row_id: row_id.to_owned(),
            features: features
                .iter()
                .map(|(name, value)| {
                    (
                        (*name).to_owned(),
                        WireValue {
                            data_type: Some(value.clone()),
                        },
                    )
                })
                .collect(),
        }
    }

    // ------------------------------------------------------- vectorised shape

    /// Order is the configuration's, not the request's: the two rows below send their
    /// features in opposite orders and must produce the same layout.
    #[test]
    fn a_vectorised_config_concatenates_in_the_configured_order() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{
                    "feature_transformations": {
                        "a": {"type": "identity", "default_val": [0.0]},
                        "b": {"type": "identity", "default_val": [0.0]}
                    },
                    "feature_order": ["a", "b"]
                }"#,
            )),
            &[spec("features", ElementType::F32, Some(2))],
        )
        .expect("2 = 1 + 1");

        let batch = builder
            .build(&[
                row(
                    "r1",
                    &[("a", Value::FloatValue(1.0)), ("b", Value::FloatValue(2.0))],
                ),
                row(
                    "r2",
                    &[("b", Value::FloatValue(4.0)), ("a", Value::FloatValue(3.0))],
                ),
            ])
            .expect("both rows are complete");

        assert_eq!(batch.rows(), 2);
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0])
        );
    }

    /// 0.1 has no exact f32. Going through the f32 path and widening would give
    /// 0.10000000149011612, so this is the test that the f64 path is real.
    #[test]
    fn a_double_vector_keeps_its_precision() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{
                    "data_type": "double",
                    "feature_transformations": {"a": {"type": "identity", "default_val": [0.0]}},
                    "feature_order": ["a"]
                }"#,
            )),
            &[spec("features", ElementType::F64, Some(1))],
        )
        .expect("one double input");

        let batch = builder
            .build(&[row("r1", &[("a", Value::DoubleValue(0.1))])])
            .expect("one row");
        assert_eq!(
            batch.get("features").expect("features").data,
            FeatureData::F64(vec![0.1])
        );
    }

    #[test]
    fn a_vector_of_the_wrong_width_is_refused_with_both_numbers() {
        let err = InputBuilder::resolve(
            Some(config(r#"{"feature_order": ["a", "b"]}"#)),
            &[spec("features", ElementType::F32, Some(8))],
        )
        .expect_err("8 != 2");
        let m = err.to_string();
        assert!(m.contains('8') && m.contains('2'), "got {m}");
    }

    #[test]
    fn a_vectorised_config_against_a_multi_input_model_is_refused() {
        let err = InputBuilder::resolve(
            Some(config(r#"{"feature_order": ["a"]}"#)),
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
            Some(config(r#"{"data_type": "float", "feature_order": ["a"]}"#)),
            &[spec("features", ElementType::F64, Some(1))],
        )
        .expect_err("FP32 is not FP64");
        let m = err.to_string();
        assert!(m.contains("FP32") && m.contains("FP64"), "got {m}");
    }

    // ------------------------------------------------------------ named shape

    #[test]
    fn named_inputs_are_fed_by_the_feature_of_the_same_name() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{
                    "feature_transformations": {
                        "age": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.0]}
                    },
                    "model_inputs": {
                        "age": {"data_type": "float"},
                        "city": {"data_type": "string"}
                    }
                }"#,
            )),
            &[
                spec("age", ElementType::F32, Some(1)),
                spec("city", ElementType::Str, Some(1)),
            ],
        )
        .expect("both inputs are declared");

        let batch = builder
            .build(&[row(
                "r1",
                &[
                    ("age", Value::FloatValue(50.0)),
                    ("city", Value::StringValue("london".into())),
                ],
            )])
            .expect("one row");

        assert_eq!(
            batch.get("age").expect("age").data,
            FeatureData::F32(vec![0.5]),
            "the transformation applies to a float input"
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
            Some(config(
                r#"{
                    "feature_transformations": {
                        "city": {"type": "one_hot_encoding", "categories": ["a", "b", "c"], "default_val": [0, 0, 0]}
                    },
                    "model_inputs": {"city": {"data_type": "float_array"}}
                }"#,
            )),
            &[spec("city", ElementType::F32, Some(3))],
        )
        .expect("three categories, three values");

        let batch = builder
            .build(&[
                row("r1", &[("city", Value::StringValue("b".into()))]),
                row("r2", &[]),
            ])
            .expect("an absent feature uses its default");

        assert_eq!(
            batch.get("city").expect("city").data,
            FeatureData::F32(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(batch.get("city").expect("city").width(), 3);
    }

    #[test]
    fn an_undeclared_model_input_is_refused_at_startup() {
        let err = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"age": {"data_type": "float"}}}"#,
            )),
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
            Some(config(
                r#"{"model_inputs": {"age": {"data_type": "float"}, "ghost": {"data_type": "float"}}}"#,
            )),
            &[spec("age", ElementType::F32, Some(1))],
        )
        .expect_err("a typo in an input name must not pass");
        assert!(err.to_string().contains("ghost"), "got {err}");
    }

    #[test]
    fn a_declared_type_that_disagrees_with_the_model_is_refused() {
        let err = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"tags": {"data_type": "float"}}}"#,
            )),
            &[spec("tags", ElementType::Str, Some(1))],
        )
        .expect_err("FP32 is not BYTES");
        let m = err.to_string();
        assert!(m.contains("FP32") && m.contains("STRING"), "got {m}");
    }

    #[test]
    fn a_scalar_declaration_against_a_wide_input_points_at_the_array_spelling() {
        let err = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"city": {"data_type": "float"}}}"#,
            )),
            &[spec("city", ElementType::F32, Some(3))],
        )
        .expect_err("one value per row cannot fill three");
        let m = err.to_string();
        assert!(m.contains("float_array"), "the fix must be named: {m}");
    }

    #[test]
    fn a_transformation_wider_than_its_input_is_refused_with_both_numbers() {
        let err = InputBuilder::resolve(
            Some(config(
                r#"{
                    "feature_transformations": {
                        "city": {"type": "one_hot_encoding", "categories": ["a", "b", "c"], "default_val": [0, 0, 0]}
                    },
                    "model_inputs": {"city": {"data_type": "float_array"}}
                }"#,
            )),
            &[spec("city", ElementType::F32, Some(5))],
        )
        .expect_err("3 != 5");
        let m = err.to_string();
        assert!(m.contains('3') && m.contains('5'), "got {m}");
    }

    #[test]
    fn a_wrongly_typed_feature_for_a_verbatim_input_names_the_row() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"tags": {"data_type": "string"}}}"#,
            )),
            &[spec("tags", ElementType::Str, Some(1))],
        )
        .expect("one text input");

        let err = builder
            .build(&[row("r7", &[("tags", Value::FloatValue(1.0))])])
            .expect_err("a float is not text");
        let m = err.to_string();
        assert!(
            m.contains("r7") && m.contains("a float"),
            "the error must name the row and what arrived: {m}"
        );
    }

    #[test]
    fn an_absent_text_feature_becomes_the_empty_string() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"tags": {"data_type": "string"}}}"#,
            )),
            &[spec("tags", ElementType::Str, Some(1))],
        )
        .expect("one text input");

        let batch = builder
            .build(&[row("r1", &[])])
            .expect("a partial row scores");
        assert_eq!(
            batch.get("tags").expect("tags").data,
            FeatureData::Str(vec![String::new()])
        );
    }

    #[test]
    fn an_absent_long_feature_becomes_zero() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"code": {"data_type": "long"}}}"#,
            )),
            &[spec("code", ElementType::I64, Some(1))],
        )
        .expect("one integer input");

        let batch = builder
            .build(&[row("r1", &[])])
            .expect("a partial row scores");
        assert_eq!(
            batch.get("code").expect("code").data,
            FeatureData::I64(vec![0])
        );
    }

    #[test]
    fn an_integer_feature_widens_into_a_long_input() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"code": {"data_type": "long"}}}"#,
            )),
            &[spec("code", ElementType::I64, Some(1))],
        )
        .expect("one integer input");

        let batch = builder
            .build(&[row("r1", &[("code", Value::IntegerValue(7))])])
            .expect("an integer is a long");
        assert_eq!(
            batch.get("code").expect("code").data,
            FeatureData::I64(vec![7])
        );
    }

    #[test]
    fn an_int_input_takes_a_32_bit_integer_feature() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"code": {"data_type": "int"}}}"#,
            )),
            &[spec("code", ElementType::I32, Some(1))],
        )
        .expect("one 32-bit integer input");

        let batch = builder
            .build(&[
                row("r1", &[("code", Value::IntegerValue(7))]),
                row("r2", &[]),
            ])
            .expect("an absent code defaults to zero");
        assert_eq!(
            batch.get("code").expect("code").data,
            FeatureData::I32(vec![7, 0])
        );
    }

    /// Narrowing is the one direction that can lose the value: a category code that
    /// silently wrapped would be a wrong prediction rather than an error. Widening the
    /// other way is exact, which is why the `long` input above takes an integer.
    #[test]
    fn a_long_feature_is_refused_for_an_int_input() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"code": {"data_type": "int"}}}"#,
            )),
            &[spec("code", ElementType::I32, Some(1))],
        )
        .expect("one 32-bit integer input");

        let err = builder
            .build(&[row(
                "r4",
                &[("code", Value::LongValue(i64::from(i32::MAX) + 1))],
            )])
            .expect_err("a long may not fit");
        let m = err.to_string();
        assert!(
            m.contains("r4") && m.contains("INT32") && m.contains("a long"),
            "the error must name the row, the input type and what arrived: {m}"
        );
    }

    #[test]
    fn an_int_array_input_takes_a_whole_vector_per_row() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"codes": {"data_type": "int_array"}}}"#,
            )),
            &[spec("codes", ElementType::I32, Some(2))],
        )
        .expect("two codes per row");

        let batch = builder
            .build(&[row(
                "r1",
                &[(
                    "codes",
                    Value::IntegerArray(hushar::hushar_proto::IntegerArray { values: vec![3, 4] }),
                )],
            )])
            .expect("one row");
        assert_eq!(
            batch.get("codes").expect("codes").data,
            FeatureData::I32(vec![3, 4])
        );
    }

    // ------------------------------------------------------ ragged and widths

    #[test]
    fn a_ragged_batch_is_refused_naming_the_row() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"tags": {"data_type": "string_array"}}}"#,
            )),
            &[spec("tags", ElementType::Str, Some(2))],
        )
        .expect("two strings per row");

        let err = builder
            .build(&[
                row(
                    "r1",
                    &[(
                        "tags",
                        Value::StringArray(StringArray {
                            values: vec!["a".into(), "b".into()],
                        }),
                    )],
                ),
                row(
                    "r2",
                    &[(
                        "tags",
                        Value::StringArray(StringArray {
                            values: vec!["c".into()],
                        }),
                    )],
                ),
            ])
            .expect_err("a batch cannot be ragged");
        let m = err.to_string();
        assert!(m.contains("r2") && m.contains('2'), "got {m}");
    }

    #[test]
    fn a_dynamic_width_with_nothing_to_size_it_is_refused_at_startup() {
        let err = InputBuilder::resolve(
            Some(config(
                r#"{"model_inputs": {"tags": {"data_type": "string_array"}}}"#,
            )),
            &[spec("tags", ElementType::Str, None)],
        )
        .expect_err("an absent feature would have no shape to fill");
        let m = err.to_string();
        assert!(m.contains("tags") && m.contains("dynamic"), "got {m}");
    }

    #[test]
    fn a_dynamic_width_is_sized_by_the_transformation() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{
                    "feature_transformations": {
                        "city": {"type": "one_hot_encoding", "categories": ["a", "b"], "default_val": [0, 0]}
                    },
                    "model_inputs": {"city": {"data_type": "float_array"}}
                }"#,
            )),
            &[spec("city", ElementType::F32, None)],
        )
        .expect("the transformation's width is the input's");

        let batch = builder.build(&[row("r1", &[])]).expect("default fills it");
        assert_eq!(batch.get("city").expect("city").width(), 2);
    }

    // ---------------------------------------------------------- no config

    #[test]
    fn without_a_configuration_every_input_takes_its_like_named_feature() {
        let builder = InputBuilder::resolve(
            None,
            &[
                spec("numbers", ElementType::F32, Some(2)),
                spec("tags", ElementType::Str, Some(1)),
            ],
        )
        .expect("a signature is enough");

        let batch = builder
            .build(&[row(
                "r1",
                &[
                    (
                        "numbers",
                        Value::FloatArray(FloatArray {
                            values: vec![1.5, 2.5],
                        }),
                    ),
                    ("tags", Value::StringValue("beta".into())),
                ],
            )])
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
        let err = InputBuilder::resolve(None, &[]).expect_err("nothing to feed");
        assert!(err.to_string().contains("no inputs"), "got {err}");
    }

    // ------------------------------------------------------------- describing

    #[test]
    fn the_banner_says_where_each_input_comes_from() {
        let builder = InputBuilder::resolve(
            Some(config(
                r#"{
                    "model_inputs": {
                        "age": {"data_type": "float"},
                        "tags": {"data_type": "string"}
                    }
                }"#,
            )),
            &[
                spec("age", ElementType::F32, Some(1)),
                spec("tags", ElementType::Str, Some(1)),
            ],
        )
        .expect("both declared");

        let described = builder.describe();
        assert!(described.contains("age FP32[1]"), "got {described}");
        assert!(described.contains("transformed"), "got {described}");
        assert!(described.contains("verbatim"), "got {described}");
    }
}
