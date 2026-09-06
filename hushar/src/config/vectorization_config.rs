// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! How a request's named features become a model's inputs, declared rather than guessed.
//!
//! Two shapes, and a configuration is exactly one of them. Which one is decided by
//! the field that is present, so there is no mode to spell and no mode to spell wrong:
//!
//! **Vectorised** — one model input, every feature transformed and concatenated in
//! `feature_order`. Strict tabular data: `rows × width` of one float type.
//!
//! ```json
//! {
//!   "data_type": "float",
//!   "feature_transformations": { "age": { "type": "identity", "default_val": [0.0] } },
//!   "feature_order": ["age", "city"]
//! }
//! ```
//!
//! **Named** — one model input per feature, each with its own declared type. This is
//! the shape that can carry text into the graph, or hand a model raw numbers to
//! normalise itself.
//!
//! ```json
//! {
//!   "feature_transformations": { "age": { "type": "standardization32", ... } },
//!   "model_inputs": {
//!     "age":  { "data_type": "float" },
//!     "city": { "data_type": "string" }
//!   }
//! }
//! ```
//!
//! An earlier version derived the mapping by reading the model's signature and
//! matching input names against feature names. That worked, but every way the guess
//! could be ambiguous needed its own error. A model file does not record the schema of
//! the data it was trained on, so the schema is asked for instead of guessed.

use crate::inference::batch::ElementType;
use crate::inference::transformations::{
    Embedding, Identity, MinMaxScaling32, MinMaxScaling64, OneHotEncoding, Standardization32,
    Standardization64, Transformation,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// What one named model input carries.
///
/// The scalar and array spellings differ only in expected width, and that is the
/// point: `float` declares one value per row, so a transformation producing five is
/// a configuration error caught at startup rather than a shape that happens to work
/// until a one-hot encoding gains a category.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputDataType {
    Float,
    Double,
    Int,
    Long,
    String,
    FloatArray,
    DoubleArray,
    IntArray,
    LongArray,
    StringArray,
}

impl InputDataType {
    /// The element type the engine sees.
    pub fn element_type(&self) -> ElementType {
        match self {
            InputDataType::Float | InputDataType::FloatArray => ElementType::F32,
            InputDataType::Double | InputDataType::DoubleArray => ElementType::F64,
            InputDataType::Int | InputDataType::IntArray => ElementType::I32,
            InputDataType::Long | InputDataType::LongArray => ElementType::I64,
            InputDataType::String | InputDataType::StringArray => ElementType::Str,
        }
    }

    /// Whether more than one value per row is expected.
    pub fn is_array(&self) -> bool {
        matches!(
            self,
            InputDataType::FloatArray
                | InputDataType::DoubleArray
                | InputDataType::IntArray
                | InputDataType::LongArray
                | InputDataType::StringArray
        )
    }

    /// The spelling used in configuration, so an error names what the operator wrote.
    pub fn name(&self) -> &'static str {
        match self {
            InputDataType::Float => "float",
            InputDataType::Double => "double",
            InputDataType::Int => "int",
            InputDataType::Long => "long",
            InputDataType::String => "string",
            InputDataType::FloatArray => "float_array",
            InputDataType::DoubleArray => "double_array",
            InputDataType::IntArray => "int_array",
            InputDataType::LongArray => "long_array",
            InputDataType::StringArray => "string_array",
        }
    }
}

impl std::fmt::Display for InputDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One entry of `model_inputs`.
///
/// A struct holding a single field rather than a bare `InputDataType`, so that
/// per-input options can be added later without rewriting every configuration.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInput {
    pub data_type: InputDataType,
}

/// The float type of a vectorised batch.
///
/// `double` exists because a tree ensemble is not a continuous function: a feature
/// value nudged across a split by a narrowing cast does not shift the prediction
/// slightly, it selects a different leaf.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorType {
    #[default]
    Float,
    Double,
}

impl VectorType {
    pub fn element_type(&self) -> ElementType {
        match self {
            VectorType::Float => ElementType::F32,
            VectorType::Double => ElementType::F64,
        }
    }
}

/// How features reach the model.
///
/// Variants:
/// - `Vectorised` — one input, carrying every feature concatenated in `feature_order`.
/// - `Named` — one input per named feature, each with its own type. Held in a
///   `BTreeMap` so the startup banner and every error list inputs the same way twice
///   running, which a `HashMap` would not.
#[derive(Clone, Debug, PartialEq)]
pub enum InputMode {
    Vectorised {
        feature_order: Vec<String>,
        data_type: VectorType,
    },
    Named {
        model_inputs: BTreeMap<String, InputDataType>,
    },
}

/// Configuration for turning a request's features into model inputs.
///
/// Fields:
/// - `feature_transformations` — feature name to the transformation applied to it.
/// - `mode` — which of the two shapes this configuration is.
///
/// After [`VectorizationConfig::from_json`] this is not a literal reading of the JSON: a
/// feature named in `feature_order` with no transformation has been given an identity
/// one, so `vector_width` can be answered without a model. Named-mode defaults cannot be
/// filled here because an array input's width comes from the model, so those are filled
/// when the configuration meets the signature.
#[derive(Debug)]
pub struct VectorizationConfig {
    pub feature_transformations: HashMap<String, Box<dyn Transformation>>,
    pub mode: InputMode,
}

/// Every available transformation, for deserialization.
///
/// The tag lives in the JSON as `"type"`, which is what lets a single map hold
/// transformations that take different parameters.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TransformationType {
    Embedding(Embedding),
    Identity(Identity),
    MinMaxScaling32(MinMaxScaling32),
    MinMaxScaling64(MinMaxScaling64),
    OneHotEncoding(OneHotEncoding),
    Standardization32(Standardization32),
    Standardization64(Standardization64),
}

impl TransformationType {
    fn into_transformation(self) -> Box<dyn Transformation> {
        match self {
            TransformationType::Embedding(t) => Box::new(t),
            TransformationType::Identity(t) => Box::new(t),
            TransformationType::MinMaxScaling32(t) => Box::new(t),
            TransformationType::MinMaxScaling64(t) => Box::new(t),
            TransformationType::OneHotEncoding(t) => Box::new(t),
            TransformationType::Standardization32(t) => Box::new(t),
            TransformationType::Standardization64(t) => Box::new(t),
        }
    }
}

/// The JSON as written, before either shape has been established.
///
/// Both shapes are read into one struct and resolved by hand rather than by an
/// untagged enum, because serde's failure for an untagged enum is "data did not
/// match any variant", which tells an operator nothing about which field to fix.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVectorizationConfig {
    #[serde(default)]
    feature_transformations: HashMap<String, TransformationType>,
    feature_order: Option<Vec<String>>,
    model_inputs: Option<BTreeMap<String, ModelInput>>,
    data_type: Option<VectorType>,
}

impl VectorizationConfig {
    /// Reads a configuration and applies the defaults it can apply without a model.
    ///
    /// A feature listed in `feature_order` with no transformation is given an identity
    /// one here, rather than at use, so that [`VectorizationConfig::vector_width`] can
    /// be answered without a model.
    pub fn from_json(json: &str) -> Result<Self, crate::inference::InferenceError> {
        let raw: RawVectorizationConfig = serde_json::from_str(json)?;

        let mut feature_transformations: HashMap<String, Box<dyn Transformation>> = raw
            .feature_transformations
            .into_iter()
            .map(|(name, t)| (name, t.into_transformation()))
            .collect();

        let mode = match (raw.feature_order, raw.model_inputs) {
            (Some(_), Some(_)) => {
                return Err("a vectorization configuration declares either \
                     \"feature_order\", for one concatenated input, or \"model_inputs\", \
                     for one input per feature -- not both. Remove whichever does not \
                     describe this model."
                    .into());
            }
            (None, None) => {
                return Err("a vectorization configuration must declare either \
                     \"feature_order\", to concatenate every feature into one model \
                     input, or \"model_inputs\", to feed one named input per feature."
                    .into());
            }
            (Some(feature_order), None) => {
                for name in &feature_order {
                    feature_transformations
                        .entry(name.clone())
                        .or_insert_with(default_identity);
                }

                InputMode::Vectorised {
                    feature_order,
                    data_type: raw.data_type.unwrap_or_default(),
                }
            }
            (None, Some(model_inputs)) => {
                if raw.data_type.is_some() {
                    return Err("\"data_type\" sets the element type of the single \
                         concatenated input, so it belongs with \"feature_order\". With \
                         \"model_inputs\", each input declares its own data_type."
                        .into());
                }
                InputMode::Named {
                    model_inputs: model_inputs
                        .into_iter()
                        .map(|(name, input)| (name, input.data_type))
                        .collect(),
                }
            }
        };

        let config = VectorizationConfig {
            feature_transformations,
            mode,
        };
        config.validate()?;
        Ok(config)
    }

    /// Every check that needs only this configuration.
    ///
    /// What needs the model as well -- widths, and whether the declared inputs are the
    /// ones the model actually has -- is checked when the two meet, in
    /// `InputBuilder::resolve`.
    ///
    /// A transformation produces floats, so one configured for a non-float input could
    /// only be applied by rounding or by inventing text. That is refused here rather
    /// than converted quietly.
    pub fn validate(&self) -> Result<(), crate::inference::InferenceError> {
        for (name, transformation) in &self.feature_transformations {
            transformation.validate(name)?;
        }

        match &self.mode {
            InputMode::Vectorised { feature_order, .. } => {
                if feature_order.is_empty() {
                    return Err("\"feature_order\" is empty, so there is nothing to \
                         concatenate into the model's input"
                        .into());
                }
                let mut seen = std::collections::HashSet::new();
                for name in feature_order {
                    if !seen.insert(name) {
                        return Err(format!(
                            "feature {name:?} appears twice in feature_order, which \
                             would place it in the vector twice"
                        )
                        .into());
                    }
                }
                for name in self.feature_transformations.keys() {
                    if !feature_order.contains(name) {
                        return Err(format!(
                            "feature {name:?} has a transformation but is not in \
                             feature_order, so it would never reach the model. Add it \
                             to feature_order or remove the transformation."
                        )
                        .into());
                    }
                }
            }
            InputMode::Named { model_inputs } => {
                if model_inputs.is_empty() {
                    return Err("\"model_inputs\" is empty, so no model input would be \
                         fed"
                    .into());
                }
                for (name, data_type) in model_inputs {
                    if !data_type.element_type().is_float()
                        && self.feature_transformations.contains_key(name)
                    {
                        return Err(format!(
                            "input {name:?} is declared {data_type}, but it has a \
                             transformation. Transformations produce floats, so a \
                             {data_type} input is passed through from the request \
                             verbatim. Remove the transformation, or declare the input \
                             as float or double."
                        )
                        .into());
                    }
                }
                for name in self.feature_transformations.keys() {
                    if !model_inputs.contains_key(name) {
                        return Err(format!(
                            "feature {name:?} has a transformation but is not declared \
                             in model_inputs, so it would never reach the model"
                        )
                        .into());
                    }
                }
            }
        }
        Ok(())
    }

    /// Elements the concatenated input carries per row.
    ///
    /// Answerable without a model, which is what makes it useful for reporting and
    /// for checking a configuration on its own. The builder computes the same sum
    /// through [`vector_width`] after it has taken ownership of the transformations.
    #[allow(dead_code)]
    pub fn vector_width(&self) -> Result<usize, crate::inference::InferenceError> {
        let InputMode::Vectorised { feature_order, .. } = &self.mode else {
            return Err(
                "this configuration feeds named inputs, so it has no single \
                 concatenated width"
                    .into(),
            );
        };
        vector_width(&self.feature_transformations, feature_order)
    }
}

/// Elements a concatenated input carries per row.
///
/// Each transformation's default vector is exactly as wide as its output -- that is
/// what makes a default usable when a feature is absent -- so the total is the sum of
/// those widths, taken in `feature_order`. Summing rather than counting is the whole
/// point: a one-hot feature is wider than one value.
///
/// A free function because the builder needs the same answer after it has taken
/// ownership of the transformations, and two implementations of one sum would be two
/// chances to disagree about a model's input width.
pub(crate) fn vector_width(
    transformations: &HashMap<String, Box<dyn Transformation>>,
    feature_order: &[String],
) -> Result<usize, crate::inference::InferenceError> {
    let mut width = 0usize;
    for name in feature_order {
        let transformation = transformations
            .get(name)
            .ok_or_else(|| format!("feature {name:?} in feature_order has no transformation"))?;
        width = width
            .checked_add(transformation.get_default_val().len())
            .ok_or("the configured feature widths overflow")?;
    }
    Ok(width)
}

/// The transformation a feature gets when configuration names none.
///
/// One value per row, zero when the feature is absent. A vectorised batch of a
/// feature that was never configured is then a column of zeros rather than a
/// startup failure, which is what makes `feature_transformations` optional.
fn default_identity() -> Box<dyn Transformation> {
    Box::new(Identity {
        default_val: vec![0.0],
    })
}

/// The identity transformation for an input of a known width.
///
/// Used where the width comes from the model rather than from configuration, which
/// is every array input with no transformation of its own.
pub(crate) fn identity_of_width(width: usize) -> Box<dyn Transformation> {
    Box::new(Identity {
        default_val: vec![0.0; width],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vectorised(json: &str) -> VectorizationConfig {
        VectorizationConfig::from_json(json).expect("a valid vectorised config")
    }

    // ------------------------------------------------------- choosing a shape

    #[test]
    fn feature_order_selects_the_vectorised_shape() {
        let config = vectorised(
            r#"{
                "feature_transformations": {
                    "age": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                    "city": {"type": "one_hot_encoding", "categories": ["a", "b", "c"], "default_val": [0, 0, 0]}
                },
                "feature_order": ["age", "city"]
            }"#,
        );
        assert_eq!(
            config.mode,
            InputMode::Vectorised {
                feature_order: vec!["age".into(), "city".into()],
                data_type: VectorType::Float,
            }
        );
        assert_eq!(config.vector_width().expect("width"), 4, "1 + 3");
    }

    #[test]
    fn model_inputs_selects_the_named_shape() {
        let config = vectorised(
            r#"{
                "feature_transformations": {
                    "age": {"type": "identity", "default_val": [0.0]}
                },
                "model_inputs": {
                    "age": {"data_type": "float"},
                    "city": {"data_type": "string"}
                }
            }"#,
        );
        let InputMode::Named { model_inputs } = &config.mode else {
            panic!("model_inputs must select the named shape");
        };
        assert_eq!(model_inputs["age"], InputDataType::Float);
        assert_eq!(model_inputs["city"], InputDataType::String);
    }

    #[test]
    fn declaring_both_shapes_is_refused_naming_both() {
        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {},
                "feature_order": ["age"],
                "model_inputs": {"age": {"data_type": "float"}}
            }"#,
        )
        .expect_err("two shapes cannot both be right");
        let m = err.to_string();
        assert!(
            m.contains("feature_order") && m.contains("model_inputs"),
            "the error must name both fields: {m}"
        );
    }

    #[test]
    fn declaring_neither_shape_is_refused_naming_both() {
        let err = VectorizationConfig::from_json(r#"{"feature_transformations": {}}"#)
            .expect_err("one shape must be chosen");
        let m = err.to_string();
        assert!(
            m.contains("feature_order") && m.contains("model_inputs"),
            "the error must name the choice: {m}"
        );
    }

    /// Without `deny_unknown_fields` this would start cleanly and behave as though the
    /// field had never been written, which is the worst of both.
    #[test]
    fn a_misspelled_field_is_refused_rather_than_ignored() {
        let err = VectorizationConfig::from_json(
            r#"{"feature_transformation": {}, "feature_order": ["age"]}"#,
        )
        .expect_err("a typo must not be silently dropped");
        assert!(
            err.to_string().contains("feature_transformation"),
            "got {err}"
        );
    }

    // ------------------------------------------------------------ the f64 case

    #[test]
    fn a_double_vector_is_declared_by_data_type() {
        let config = vectorised(
            r#"{
                "feature_transformations": {"age": {"type": "identity", "default_val": [0.0]}},
                "feature_order": ["age"],
                "data_type": "double"
            }"#,
        );
        assert_eq!(
            config.mode,
            InputMode::Vectorised {
                feature_order: vec!["age".into()],
                data_type: VectorType::Double,
            }
        );
    }

    #[test]
    fn data_type_belongs_to_the_vectorised_shape_only() {
        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {},
                "model_inputs": {"age": {"data_type": "float"}},
                "data_type": "double"
            }"#,
        )
        .expect_err("a named input declares its own type");
        assert!(err.to_string().contains("data_type"), "got {err}");
    }

    // ------------------------------------------------------- default identity

    /// The whole `feature_transformations` map may be omitted, and each feature then
    /// contributes one value per row.
    #[test]
    fn a_feature_with_no_transformation_is_passed_through() {
        let config = vectorised(r#"{"feature_order": ["a", "b", "c"]}"#);
        assert_eq!(config.vector_width().expect("width"), 3);
        assert!(
            config.feature_transformations.contains_key("b"),
            "the default must have been filled in"
        );
    }

    #[test]
    fn a_declared_transformation_is_not_overwritten_by_the_default() {
        let config = vectorised(
            r#"{
                "feature_transformations": {
                    "city": {"type": "one_hot_encoding", "categories": ["a", "b"], "default_val": [0, 0]}
                },
                "feature_order": ["age", "city"]
            }"#,
        );
        assert_eq!(
            config.vector_width().expect("width"),
            3,
            "age defaults to 1, city keeps its 2"
        );
    }

    // ------------------------------------------------------------- validation

    /// Transformations produce floats, so one attached to a text input could only be
    /// applied by inventing text from a number.
    #[test]
    fn a_transformation_on_a_string_input_is_refused() {
        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {"city": {"type": "identity", "default_val": [0.0]}},
                "model_inputs": {"city": {"data_type": "string"}}
            }"#,
        )
        .expect_err("a text input is fed verbatim");
        let m = err.to_string();
        assert!(m.contains("city") && m.contains("string"), "got {m}");
    }

    #[test]
    fn a_transformation_on_a_long_input_is_refused() {
        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {"code": {"type": "identity", "default_val": [0.0]}},
                "model_inputs": {"code": {"data_type": "long"}}
            }"#,
        )
        .expect_err("rounding a float into an integer would be corruption");
        assert!(err.to_string().contains("code"), "got {err}");
    }

    #[test]
    fn a_transformation_that_could_never_run_is_refused() {
        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {"ghost": {"type": "identity", "default_val": [0.0]}},
                "feature_order": ["age"]
            }"#,
        )
        .expect_err("an unreachable transformation is a mistake");
        assert!(err.to_string().contains("ghost"), "got {err}");
    }

    #[test]
    fn a_repeated_feature_in_the_order_is_refused() {
        let err = VectorizationConfig::from_json(r#"{"feature_order": ["age", "age"]}"#)
            .expect_err("a feature cannot be in the vector twice");
        assert!(err.to_string().contains("age"), "got {err}");
    }

    #[test]
    fn an_empty_shape_is_refused_in_both_modes() {
        assert!(
            VectorizationConfig::from_json(r#"{"feature_order": []}"#).is_err(),
            "nothing to concatenate"
        );
        assert!(
            VectorizationConfig::from_json(r#"{"model_inputs": {}}"#).is_err(),
            "no input would be fed"
        );
    }

    #[test]
    fn an_invalid_transformation_is_refused_with_its_feature_named() {
        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {
                    "age": {"type": "min_max_scaling32", "min": 100.0, "max": 0.0, "default_val": [0.5]}
                },
                "feature_order": ["age"]
            }"#,
        )
        .expect_err("min must be below max");
        assert!(err.to_string().contains("age"), "got {err}");
    }

    #[test]
    fn an_unknown_transformation_type_is_refused() {
        assert!(
            VectorizationConfig::from_json(
                r#"{
                    "feature_transformations": {"age": {"type": "nonesuch", "default_val": [0.0]}},
                    "feature_order": ["age"]
                }"#,
            )
            .is_err()
        );
    }

    #[test]
    fn every_transformation_type_parses() {
        let config = vectorised(
            r#"{
                "feature_transformations": {
                    "f1": {"type": "embedding", "embeddings": {"a": [0.0, 1.0]}, "default_val": [0.0, 0.0]},
                    "f2": {"type": "identity", "default_val": [0.0]},
                    "f3": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                    "f4": {"type": "min_max_scaling64", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                    "f5": {"type": "one_hot_encoding", "categories": ["A", "B", "C"], "default_val": [0, 0, 0]},
                    "f6": {"type": "standardization32", "mean": 50.0, "std_dev": 10.0, "default_val": [0.0]},
                    "f7": {"type": "standardization64", "mean": 50.0, "std_dev": 10.0, "default_val": [0.0]}
                },
                "feature_order": ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
            }"#,
        );
        assert_eq!(config.feature_transformations.len(), 7);
        assert_eq!(
            config.vector_width().expect("width"),
            2 + 1 + 1 + 1 + 3 + 1 + 1
        );
    }

    // ----------------------------------------------------------- data types

    #[test]
    fn each_declared_type_maps_to_one_element_type() {
        for (declared, element, array) in [
            (InputDataType::Float, ElementType::F32, false),
            (InputDataType::Double, ElementType::F64, false),
            (InputDataType::Int, ElementType::I32, false),
            (InputDataType::Long, ElementType::I64, false),
            (InputDataType::String, ElementType::Str, false),
            (InputDataType::FloatArray, ElementType::F32, true),
            (InputDataType::DoubleArray, ElementType::F64, true),
            (InputDataType::IntArray, ElementType::I32, true),
            (InputDataType::LongArray, ElementType::I64, true),
            (InputDataType::StringArray, ElementType::Str, true),
        ] {
            assert_eq!(declared.element_type(), element, "{declared}");
            assert_eq!(declared.is_array(), array, "{declared}");
        }
    }

    #[test]
    fn the_config_spelling_of_every_type_deserializes() {
        for spelling in [
            "float",
            "double",
            "int",
            "long",
            "string",
            "float_array",
            "double_array",
            "int_array",
            "long_array",
            "string_array",
        ] {
            let json = format!(r#"{{"model_inputs": {{"x": {{"data_type": "{spelling}"}}}}}}"#);
            assert!(
                VectorizationConfig::from_json(&json).is_ok(),
                "{spelling} must parse"
            );
        }
    }

    #[test]
    fn a_named_config_has_no_single_width() {
        let config = vectorised(r#"{"model_inputs": {"age": {"data_type": "float"}}}"#);
        assert!(
            config.vector_width().is_err(),
            "asking for one width across named inputs is a category error"
        );
    }
}
