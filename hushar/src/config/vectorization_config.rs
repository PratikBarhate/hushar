// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! How a request's named features become a model's inputs.
//!
//! Two shapes, and a configuration is exactly one of them. Which one is decided by the
//! field that is present:
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
//! **Named** — one model input per feature, each with its own declared type, taken from
//! the request verbatim. It transforms nothing, so it declares no transformations.
//!
//! ```json
//! {
//!   "model_inputs": {
//!     "age":  { "data_type": "float" },
//!     "city": { "data_type": "string" }
//!   }
//! }
//! ```

use crate::inference::batch::ElementType;
use crate::inference::transformations::{
    Embedding, Identity, MinMaxScaling32, MinMaxScaling64, OneHotEncoding, Standardization32,
    Standardization64, Transformation,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// What one named model input carries.
///
/// The scalar and array spellings differ in expected width: `float` declares one value
/// per row, `float_array` more than one. A width that disagrees with the model is a
/// configuration error caught at startup.
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

    /// The spelling this type takes in configuration.
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInput {
    pub data_type: InputDataType,
}

/// The float type of a vectorised batch.
///
/// Accepts `float` and `double`. Defaults to `float`.
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
/// # Fields
///
/// * `Vectorised` - one input, carrying every feature transformed and concatenated in
///   `feature_order`; the only mode that transforms
/// * `Named` - one input per named feature, each with its own type, taken from the
///   request verbatim; held in a `BTreeMap`, so inputs are always reported in name order
#[derive(Debug)]
pub enum InputMode {
    Vectorised {
        feature_order: Vec<String>,
        data_type: VectorType,
        transformations: HashMap<String, Box<dyn Transformation>>,
    },
    Named {
        model_inputs: BTreeMap<String, InputDataType>,
    },
}

/// Configuration for turning a request's features into model inputs.
///
/// # Fields
///
/// * `mode` - which of the two shapes this configuration is; transformations live inside
///   `InputMode::Vectorised`, so a `Named` configuration cannot carry any, and
///   [`VectorizationConfig::from_json`] rejects the JSON that asks for it
///
/// After [`VectorizationConfig::from_json`] this is not a literal reading of the JSON: a
/// feature named in `feature_order` with no transformation has been given an identity
/// one.
#[derive(Debug)]
pub struct VectorizationConfig {
    pub mode: InputMode,
}

/// Every available transformation, for deserialization.
///
/// The variant is chosen by the JSON `type` field.
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
    /// one here, so [`VectorizationConfig::vector_width`] can be answered without a
    /// model.
    ///
    /// # Errors
    ///
    /// Returns an error if neither or both of `feature_order` and `model_inputs` are
    /// declared, if `feature_transformations` or `data_type` appear alongside
    /// `model_inputs`, or if [`VectorizationConfig::validate`] refuses the result.
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
                    transformations: feature_transformations,
                }
            }
            (None, Some(model_inputs)) => {
                if !feature_transformations.is_empty() {
                    let mut named: Vec<&str> =
                        feature_transformations.keys().map(String::as_str).collect();
                    named.sort_unstable();
                    return Err(format!(
                        "\"model_inputs\" feeds every input from the request verbatim, so \
                         a transformation would never run: {}. Remove \
                         \"feature_transformations\", or use \"feature_order\" to \
                         concatenate transformed features into one input.",
                        named.join(", ")
                    )
                    .into());
                }
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

        let config = VectorizationConfig { mode };
        config.validate()?;
        Ok(config)
    }

    /// Every check that needs only this configuration.
    ///
    /// What needs the model as well -- widths, and whether the declared inputs are the
    /// ones the model actually has -- is checked when the two meet, in
    /// `InputBuilder::resolve`.
    ///
    /// # Errors
    ///
    /// Returns an error if a transformation is itself invalid, if `feature_order` is
    /// empty or repeats a feature, if a transformation names a feature absent from
    /// `feature_order`, or if `model_inputs` is empty.
    pub fn validate(&self) -> Result<(), crate::inference::InferenceError> {
        if let InputMode::Vectorised {
            transformations, ..
        } = &self.mode
        {
            for (name, transformation) in transformations {
                transformation.validate(name)?;
            }
        }

        match &self.mode {
            InputMode::Vectorised {
                feature_order,
                transformations,
                ..
            } => {
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
                for name in transformations.keys() {
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
            }
        }
        Ok(())
    }

    /// Elements the concatenated input carries per row.
    ///
    /// # Errors
    ///
    /// Returns an error if this configuration feeds named inputs, which have no single
    /// concatenated width, if a feature in `feature_order` has no transformation, or if
    /// the widths overflow.
    #[allow(dead_code)]
    pub fn vector_width(&self) -> Result<usize, crate::inference::InferenceError> {
        let InputMode::Vectorised {
            feature_order,
            transformations,
            ..
        } = &self.mode
        else {
            return Err(
                "this configuration feeds named inputs, so it has no single \
                 concatenated width"
                    .into(),
            );
        };
        vector_width(transformations, feature_order)
    }
}

/// Elements a concatenated input carries per row.
///
/// Each transformation's default vector is exactly as wide as its output, so the total
/// is the sum of those widths taken in `feature_order` -- a sum, not a feature count,
/// because a one-hot feature is wider than one value.
///
/// # Errors
///
/// Returns an error if a feature in `feature_order` has no transformation, or if the
/// widths overflow.
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
/// One value per row, zero when the feature is absent, which is what makes
/// `feature_transformations` optional.
fn default_identity() -> Box<dyn Transformation> {
    Box::new(Identity {
        default_val: vec![0.0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transformations of a vectorised config, which is the only shape that has any.
    fn transformations_of(
        config: &VectorizationConfig,
    ) -> &HashMap<String, Box<dyn Transformation>> {
        let InputMode::Vectorised {
            transformations, ..
        } = &config.mode
        else {
            panic!("only the vectorised shape carries transformations");
        };
        transformations
    }

    /// Asserts a config is the vectorised shape with this order and element type.
    fn assert_vectorised(config: &VectorizationConfig, order: &[&str], data_type: VectorType) {
        let InputMode::Vectorised {
            feature_order,
            data_type: actual,
            ..
        } = &config.mode
        else {
            panic!("feature_order must select the vectorised shape");
        };
        assert_eq!(feature_order.as_slice(), order);
        assert_eq!(*actual, data_type);
    }

    fn vectorised(json: &str) -> VectorizationConfig {
        VectorizationConfig::from_json(json).expect("a valid vectorised config")
    }

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
        assert_vectorised(&config, &["age", "city"], VectorType::Float);
        assert_eq!(config.vector_width().expect("width"), 4, "1 + 3");
    }

    #[test]
    fn model_inputs_selects_the_named_shape() {
        let config = vectorised(
            r#"{
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

    /// Asserts a misspelled field is refused rather than silently ignored.
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

    #[test]
    fn a_double_vector_is_declared_by_data_type() {
        let config = vectorised(
            r#"{
                "feature_transformations": {"age": {"type": "identity", "default_val": [0.0]}},
                "feature_order": ["age"],
                "data_type": "double"
            }"#,
        );
        assert_vectorised(&config, &["age"], VectorType::Double);
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

    /// Asserts an omitted `feature_transformations` gives each feature one value per row.
    #[test]
    fn a_feature_with_no_transformation_is_passed_through() {
        let config = vectorised(r#"{"feature_order": ["a", "b", "c"]}"#);
        assert_eq!(config.vector_width().expect("width"), 3);
        assert!(
            transformations_of(&config).contains_key("b"),
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

    /// Asserts a transformation beside `model_inputs` is refused, whatever the declared
    /// type.
    #[test]
    fn a_transformation_alongside_model_inputs_is_refused() {
        for data_type in ["string", "float"] {
            let json = format!(
                r#"{{
                    "feature_transformations": {{"city": {{"type": "identity", "default_val": [0.0]}}}},
                    "model_inputs": {{"city": {{"data_type": "{data_type}"}}}}
                }}"#
            );
            let err = VectorizationConfig::from_json(&json)
                .expect_err("a named input is fed verbatim, whatever its type");
            let m = err.to_string();
            assert!(
                m.contains("city") && m.contains("verbatim") && m.contains("feature_order"),
                "a {data_type} input should say which feature, that it is verbatim, and \
                 what to use instead; got {m}"
            );
        }
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
        assert_eq!(transformations_of(&config).len(), 7);
        assert_eq!(
            config.vector_width().expect("width"),
            2 + 1 + 1 + 1 + 3 + 1 + 1
        );
    }

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
