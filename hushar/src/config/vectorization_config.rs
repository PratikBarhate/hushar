// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! How a request's named features become the model's input.
//!
//! Every feature named in `feature_order` is transformed and concatenated into the
//! model's one input. Strict tabular data: `rows × width` of one float type.
//!
//! ```json
//! {
//!   "data_type": "float",
//!   "feature_transformations": { "age": { "type": "identity", "default_val": [0.0] } },
//!   "feature_order": ["age", "city"]
//! }
//! ```

use crate::inference::batch::ElementType;
use crate::inference::transformations::{
    EmbeddingLookup, Identity, MinMaxScaling32, MinMaxScaling64, OneHotEncoding, Standardization32,
    Standardization64, Transformation,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    /// The element type the engine sees.
    pub fn element_type(&self) -> ElementType {
        match self {
            VectorType::Float => ElementType::F32,
            VectorType::Double => ElementType::F64,
        }
    }

    /// The spelling this type takes in configuration.
    pub fn name(&self) -> &'static str {
        match self {
            VectorType::Float => "float",
            VectorType::Double => "double",
        }
    }
}

impl std::fmt::Display for VectorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Configuration for turning a request's features into the model's input.
///
/// # Fields
///
/// * `feature_order` - the features the input carries, in the order they are
///   concatenated
/// * `data_type` - the element type of the concatenated input
/// * `transformations` - the transformation that produces each feature's values
///
/// After [`VectorizationConfig::from_json`] this is not a literal reading of the JSON: a
/// feature named in `feature_order` with no transformation has been given an identity
/// one.
#[derive(Debug)]
pub struct VectorizationConfig {
    pub feature_order: Vec<String>,
    pub data_type: VectorType,
    pub transformations: HashMap<String, Box<dyn Transformation>>,
}

/// Every available transformation, for deserialization.
///
/// The variant is chosen by the JSON `type` field.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TransformationType {
    /// `embedding` is the name this went by before it said that it is a table lookup.
    #[serde(alias = "embedding")]
    EmbeddingLookup(EmbeddingLookup),
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
            TransformationType::EmbeddingLookup(t) => Box::new(t),
            TransformationType::Identity(t) => Box::new(t),
            TransformationType::MinMaxScaling32(t) => Box::new(t),
            TransformationType::MinMaxScaling64(t) => Box::new(t),
            TransformationType::OneHotEncoding(t) => Box::new(t),
            TransformationType::Standardization32(t) => Box::new(t),
            TransformationType::Standardization64(t) => Box::new(t),
        }
    }
}

/// The JSON as written, before the defaults have been applied.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVectorizationConfig {
    #[serde(default)]
    feature_transformations: HashMap<String, TransformationType>,
    feature_order: Option<Vec<String>>,
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
    /// Returns an error if `feature_order` is not declared, or if
    /// [`VectorizationConfig::validate`] refuses the result.
    pub fn from_json(json: &str) -> Result<Self, crate::inference::InferenceError> {
        let raw: RawVectorizationConfig = serde_json::from_str(json)?;

        let Some(feature_order) = raw.feature_order else {
            return Err(
                "a vectorization configuration must declare \"feature_order\", \
                 which names the features concatenated into the model's input."
                    .into(),
            );
        };

        let mut transformations: HashMap<String, Box<dyn Transformation>> = raw
            .feature_transformations
            .into_iter()
            .map(|(name, t)| (name, t.into_transformation()))
            .collect();
        for name in &feature_order {
            transformations
                .entry(name.clone())
                .or_insert_with(default_identity);
        }

        let config = VectorizationConfig {
            feature_order,
            data_type: raw.data_type.unwrap_or_default(),
            transformations,
        };
        config.validate()?;
        Ok(config)
    }

    /// Every check that needs only this configuration.
    ///
    /// What needs the model as well -- the width, and the element type the model's input
    /// takes -- is checked when the two meet, in `InputBuilder::resolve`.
    ///
    /// # Errors
    ///
    /// Returns an error if a transformation is itself invalid, if `feature_order` is
    /// empty or repeats a feature, or if a transformation names a feature absent from
    /// `feature_order`.
    pub fn validate(&self) -> Result<(), crate::inference::InferenceError> {
        for (name, transformation) in &self.transformations {
            transformation.validate(name)?;
        }

        if self.feature_order.is_empty() {
            return Err("\"feature_order\" is empty, so there is nothing to \
                 concatenate into the model's input"
                .into());
        }

        let mut seen = HashSet::new();
        for name in &self.feature_order {
            if !seen.insert(name) {
                return Err(format!(
                    "feature {name:?} appears twice in feature_order, which would place \
                     it in the vector twice"
                )
                .into());
            }
        }

        for name in self.transformations.keys() {
            if !self.feature_order.contains(name) {
                return Err(format!(
                    "feature {name:?} has a transformation but is not in feature_order, \
                     so it would never reach the model. Add it to feature_order or \
                     remove the transformation."
                )
                .into());
            }
        }
        Ok(())
    }

    /// Elements the concatenated input carries per row.
    ///
    /// Each transformation's default vector is exactly as wide as its output, so the
    /// total is the sum of those widths taken in `feature_order` -- a sum, not a feature
    /// count, because a one-hot feature is wider than one value.
    ///
    /// # Errors
    ///
    /// Returns an error if a feature in `feature_order` has no transformation, or if the
    /// widths overflow.
    pub fn vector_width(&self) -> Result<usize, crate::inference::InferenceError> {
        let mut width = 0usize;
        for name in &self.feature_order {
            let transformation = self.transformations.get(name).ok_or_else(|| {
                format!("feature {name:?} in feature_order has no transformation")
            })?;
            width = width
                .checked_add(transformation.get_default_val().len())
                .ok_or("the configured feature widths overflow")?;
        }
        Ok(width)
    }
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

    fn vectorised(json: &str) -> VectorizationConfig {
        VectorizationConfig::from_json(json).expect("a valid vectorised config")
    }

    #[test]
    fn a_config_carries_its_feature_order_and_element_type() {
        let config = vectorised(
            r#"{
                "feature_transformations": {
                    "age": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                    "city": {"type": "one_hot_encoding", "categories": ["a", "b", "c"], "default_val": [0, 0, 0]}
                },
                "feature_order": ["age", "city"]
            }"#,
        );
        assert_eq!(config.feature_order, ["age", "city"]);
        assert_eq!(config.data_type, VectorType::Float);
        assert_eq!(config.vector_width().expect("width"), 4, "1 + 3");
    }

    #[test]
    fn a_config_without_a_feature_order_is_refused_naming_the_field() {
        let err = VectorizationConfig::from_json(r#"{"feature_transformations": {}}"#)
            .expect_err("nothing would say what the input carries");
        assert!(
            err.to_string().contains("feature_order"),
            "the error must name the field: {err}"
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
        assert_eq!(config.data_type, VectorType::Double);
    }

    /// Asserts an omitted `feature_transformations` gives each feature one value per row.
    #[test]
    fn a_feature_with_no_transformation_is_passed_through() {
        let config = vectorised(r#"{"feature_order": ["a", "b", "c"]}"#);
        assert_eq!(config.vector_width().expect("width"), 3);
        assert!(
            config.transformations.contains_key("b"),
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
    fn an_empty_feature_order_is_refused() {
        assert!(
            VectorizationConfig::from_json(r#"{"feature_order": []}"#).is_err(),
            "nothing to concatenate"
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
                    "f1": {"type": "embedding_lookup", "embeddings": {"a": [0.0, 1.0]}, "default_val": [0.0, 0.0]},
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
        assert_eq!(config.transformations.len(), 7);
        assert_eq!(
            config.vector_width().expect("width"),
            2 + 1 + 1 + 1 + 3 + 1 + 1
        );
    }

    /// A configuration written before the rename still loads.
    ///
    /// `embedding_lookup` was called `embedding`, and configurations outlive the names
    /// they were written against, so the old tag stays readable.
    #[test]
    fn the_former_name_for_embedding_lookup_still_parses() {
        let config = vectorised(
            r#"{
                "feature_transformations": {
                    "f1": {"type": "embedding", "embeddings": {"a": [0.0, 1.0]}, "default_val": [0.0, 0.0]}
                },
                "feature_order": ["f1"]
            }"#,
        );

        assert_eq!(config.transformations.len(), 1);
        assert_eq!(config.vector_width().expect("width"), 2);
    }

    /// Loading a configuration refuses a row whose width would depend on the value.
    ///
    /// Both are rejected by the transformation's own `validate`, which this reaches
    /// through `from_json`, so the mistake is caught before a request ever arrives.
    #[test]
    fn a_transformation_of_inconsistent_width_is_refused_at_load() {
        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {
                    "f1": {"type": "embedding_lookup", "embeddings": {"a": [0.0, 1.0], "b": [0.0]}, "default_val": [0.0, 0.0]}
                },
                "feature_order": ["f1"]
            }"#,
        )
        .expect_err("one vector is narrower than the rest");
        assert!(err.to_string().contains("wide"), "got {err}");

        let err = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {
                    "f1": {"type": "one_hot_encoding", "categories": ["B", "A"], "default_val": [0, 0]}
                },
                "feature_order": ["f1"]
            }"#,
        )
        .expect_err("the categories are out of order");
        assert!(err.to_string().contains("sorted"), "got {err}");
    }

    #[test]
    fn each_vector_type_maps_to_one_element_type() {
        for (declared, element, name) in [
            (VectorType::Float, ElementType::F32, "float"),
            (VectorType::Double, ElementType::F64, "double"),
        ] {
            assert_eq!(declared.element_type(), element, "{declared}");
            assert_eq!(declared.name(), name);
        }
    }

    #[test]
    fn both_vector_type_spellings_deserialize() {
        for spelling in ["float", "double"] {
            let json = format!(r#"{{"feature_order": ["a"], "data_type": "{spelling}"}}"#);
            assert!(
                VectorizationConfig::from_json(&json).is_ok(),
                "{spelling} must parse"
            );
        }
    }
}
