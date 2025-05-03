// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Vectorization configuration module to read the configurations for JSON string.

use crate::inference::transformations::{
    Embedding, Identity, MinMaxScaling32, MinMaxScaling64, OneHotEncoding, Standardization32,
    Standardization64, Transformation,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

/// Configuration for vectorizing features for machine learning models.
///
/// This struct defines which transformations should be applied to each feature
/// and in what order the features should be processed.
#[derive(Debug)]
pub struct VectorizationConfig {
    /// Mapping from feature names to their transformation implementations
    pub feature_transformations: HashMap<String, Box<dyn Transformation>>,
    /// The order in which features should be processed
    pub feature_order: Vec<String>,
}

/// Enum representing all available transformation types for serialization/deserialization.
///
/// This enum is used internally to help with serializing and deserializing the
/// dynamic `Box<dyn Transformation>` values.
#[derive(Debug, Serialize, Deserialize)]
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

/// A serializable representation of VectorizationConfig.
///
/// This struct is used as an intermediate representation for serialization and
/// deserialization, as we cannot directly serialize/deserialize trait objects.
#[derive(Debug, Serialize, Deserialize)]
struct SerializableVectorizationConfig {
    feature_transformations: HashMap<String, TransformationType>,
    feature_order: Vec<String>,
}

impl VectorizationConfig {
    /// Deserializes a JSON string into a VectorizationConfig.
    ///
    /// # Arguments
    /// * `json` - The JSON string to deserialize
    ///
    /// # Returns
    /// * `Result<VectorizationConfig, Box<dyn Error>>` - The deserialized config or an error
    pub fn from_json(json: &str) -> Result<Self, Box<dyn Error>> {
        // Deserialize from JSON to the intermediate representation
        let serializable: SerializableVectorizationConfig = serde_json::from_str(json)?;

        // Convert back to VectorizationConfig
        let mut feature_transformations = HashMap::new();

        for (key, transformation_type) in serializable.feature_transformations {
            let transformation: Box<dyn Transformation> = match transformation_type {
                TransformationType::Embedding(t) => Box::new(t),
                TransformationType::Identity(t) => Box::new(t),
                TransformationType::MinMaxScaling32(t) => Box::new(t),
                TransformationType::MinMaxScaling64(t) => Box::new(t),
                TransformationType::OneHotEncoding(t) => Box::new(t),
                TransformationType::Standardization32(t) => Box::new(t),
                TransformationType::Standardization64(t) => Box::new(t),
            };

            feature_transformations.insert(key, transformation);
        }

        let vec_config = VectorizationConfig {
            feature_transformations,
            feature_order: serializable.feature_order,
        };
        vec_config.validate().unwrap();
        Ok(vec_config)
    }

    /// Validates all transformations in the configuration.
    ///
    /// Calls the validate method on each transformation in the configuration.
    ///
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Ok if all transformations are valid, Err otherwise
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        for (name, transformation) in &self.feature_transformations {
            transformation.validate(name)?;
        }

        // Check that all features in feature_order have a transformation
        for feature in &self.feature_order {
            if !self.feature_transformations.contains_key(feature) {
                return Err(
                    format!("Feature {} in feature_order has no transformation", feature).into(),
                );
            }
        }

        // Check that all transformations are in feature_order
        for feature in self.feature_transformations.keys() {
            if !self.feature_order.contains(feature) {
                return Err(format!(
                    "Feature {} has a transformation but is not in feature_order",
                    feature
                )
                .into());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hushar_proto::hushar::data_type::DataType;

    #[test]
    fn test_from_json_with_valid_json() {
        // Arrange
        let json = r#"{
            "feature_transformations": {
                "age": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                "name": {"type": "one_hot_encoding", "categories": ["Alice", "Bob", "Charlie"], "default_val": [0, 0, 0]}
            },
            "feature_order": ["age", "name"]
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);

        // Assert
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.feature_order, vec!["age", "name"]);
        assert_eq!(config.feature_transformations.len(), 2);
        assert!(config.feature_transformations.contains_key("age"));
        assert!(config.feature_transformations.contains_key("name"));
    }

    #[test]
    fn test_from_json_with_all_transformation_types() {
        // Arrange
        let json = r#"{
            "feature_transformations": {
                "feat1": {"type": "embedding", "embeddings": {"hello" : [0.0, 1.0], "world" : [1.0, 0.0], "test" : [1.0, 1.0]}, "default_val": [0.0, 0.0]},
                "feat2": {"type": "identity", "default_val": [0.0]},
                "feat3": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                "feat4": {"type": "min_max_scaling64", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                "feat5": {"type": "one_hot_encoding", "categories": ["A", "B", "C"], "default_val": [0, 0, 0]},
                "feat6": {"type": "standardization32", "mean": 50.0, "std_dev": 10.0, "default_val": [0.0]},
                "feat7": {"type": "standardization64", "mean": 50.0, "std_dev": 10.0, "default_val": [0.0]}
            },
            "feature_order": ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "feat7"]
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);

        // Assert
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.feature_order.len(), 7);
        assert_eq!(config.feature_transformations.len(), 7);
    }

    #[test]
    fn test_from_json_with_invalid_json_format() {
        // Arrange
        let json = r#"{
            "feature_transformations": {
                "age": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.5]},
            }, // Invalid trailing comma
            "feature_order": ["age", "name"]
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_with_unknown_transformation_type() {
        // Arrange
        let json = r#"{
            "feature_transformations": {
                "age": {"type": "invalid_transformation_type", "min": 0.0, "max": 100.0, "default_val": [0.5]}
            },
            "feature_order": ["age"]
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_with_missing_fields() {
        // Arrange
        let json = r#"{
            "feature_transformations": {
                "age": {"type": "min_max_scaling32", "min": 100.0, "default_val": [0.5]}
            },
            "feature_order": ["age"]
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_with_valid_config() {
        // Arrange
        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "age".to_string(),
            Box::new(MinMaxScaling32 {
                min: 0.0,
                max: 100.0,
                default_val: vec![0.5],
            }) as Box<dyn Transformation>,
        );
        let feature_order = vec!["age".to_string()];

        let config = VectorizationConfig {
            feature_transformations,
            feature_order,
        };

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_with_feature_in_order_but_no_transformation() {
        // Arrange
        let feature_transformations = HashMap::new();
        let feature_order = vec!["age".to_string()];

        let config = VectorizationConfig {
            feature_transformations,
            feature_order,
        };

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Feature age in feature_order has no transformation"));
    }

    #[test]
    fn test_validate_with_transformation_but_not_in_order() {
        // Arrange
        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "age".to_string(),
            Box::new(MinMaxScaling32 {
                min: 0.0,
                max: 100.0,
                default_val: vec![0.5],
            }) as Box<dyn Transformation>,
        );
        let feature_order = vec![];

        let config = VectorizationConfig {
            feature_transformations,
            feature_order,
        };

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Feature age has a transformation but is not in feature_order"));
    }

    #[test]
    fn test_validate_with_invalid_transformation() {
        // Arrange - Create a transformation that will fail validation
        #[derive(Clone, Debug, Deserialize, Serialize)]
        struct InvalidTransformation {
            default_val: Vec<f32>,
        }

        impl Transformation for InvalidTransformation {
            fn transform(&self, _value: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
                Ok(vec![0.0])
            }

            fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
                Err(format!("Invalid transformation for feature {}", name).into())
            }

            fn get_default_val(&self) -> &[f32] {
                &self.default_val
            }
        }

        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "test".to_string(),
            Box::new(InvalidTransformation {
                default_val: vec![0.4, 0.8],
            }) as Box<dyn Transformation>,
        );
        let feature_order = vec!["test".to_string()];

        let config = VectorizationConfig {
            feature_transformations,
            feature_order,
        };

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid transformation for feature test"));
    }

    #[test]
    fn test_from_json_integration_with_validate() {
        // Arrange
        let json = r#"{
            "feature_transformations": {
                "age": {"type": "min_max_scaling32", "min": 0.0, "max": 100.0, "default_val": [0.5]},
                "name": {"type": "one_hot_encoding", "categories": ["Alice", "Bob", "Charlie"], "default_val": [0, 0, 0]}
            },
            "feature_order": ["age", "name"]
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);
        assert!(result.is_ok());
        let config = result.unwrap();
        let validation_result = config.validate();

        // Assert
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_complex_config_deserialization() {
        // Arrange
        let json = r#"{
            "feature_transformations": {
                "numerical_feature1": {"type": "min_max_scaling32", "min": -10.0, "max": 10.0, "default_val": [0.0]},
                "numerical_feature2": {"type": "standardization64", "mean": 15.5, "std_dev": 3.2, "default_val": [0.0]},
                "categorical_feature1": {"type": "one_hot_encoding", "categories": ["Red", "Green", "Blue", "Yellow"], "default_val": [0, 0, 0, 0]},
                "text_feature": {"type": "embedding", "embeddings": {"hello" : [0.0, 1.0], "world" : [1.0, 0.0], "test" : [1.0, 1.0]}, "default_val": [0.0, 0.0]},
                "passthrough_feature": {"type": "identity", "default_val": [1.0]}
            },
            "feature_order": ["numerical_feature1", "numerical_feature2", "categorical_feature1", "text_feature", "passthrough_feature"]
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);

        // Assert
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.feature_order.len(), 5);
        assert_eq!(config.feature_transformations.len(), 5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_empty_config() {
        // Arrange
        let json = r#"{
            "feature_transformations": {},
            "feature_order": []
        }"#;

        // Act
        let result = VectorizationConfig::from_json(json);

        // Assert
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.feature_order.len(), 0);
        assert_eq!(config.feature_transformations.len(), 0);
        assert!(config.validate().is_ok());
    }
}
