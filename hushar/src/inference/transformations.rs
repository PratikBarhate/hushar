// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Feature transformation implementations for converting different data types into ML model inputs.
//!
//! This module provides various transformation strategies for feature preprocessing, including:
//! - Embedding: Maps string values to pre-trained vector embeddings
//! - Identity: Passes through numerical values with minimal transformation
//! - MinMaxScaling: Scales numerical values to a specified range
//! - OneHotEncoding: Converts categorical values to one-hot encoded vectors
//! - Standardization: Standardizes numerical values using mean and standard deviation

use hushar_proto::hushar::data_type::DataType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

/// Defines the interface for feature transformation implementations.
///
/// Implementations of this trait can transform various data types into
/// vectors of f32 values suitable for machine learning model input.
pub trait Transformation: std::fmt::Debug + Send + Sync {
    /// Returns the default value to use when transformation is not possible.
    fn get_default_val(&self) -> &[f32];

    /// Transforms the provided feature value into a vector of f32 values.
    ///
    /// # Arguments
    /// * `feat_val` - The feature value to transform
    ///
    /// # Returns
    /// * `Result<Vec<f32>, Box<dyn Error>>` - The transformed feature value or an error
    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>>;

    /// Validates the transformation configuration.
    ///
    /// # Arguments
    /// * `name` - The name of the feature being validated
    ///
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Ok if valid, Err otherwise with a descriptive message
    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>>;
}

/// Maps string values to vector embeddings using a pre-defined embedding table.
///
/// If the string value is not found in the embedding table, the default value is used.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Embedding {
    /// Default value to use when a string is not found in the embeddings
    pub default_val: Vec<f32>,
    /// Mapping from string values to their vector embeddings
    pub embeddings: HashMap<String, Vec<f32>>,
}

impl Transformation for Embedding {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
        match feat_val {
            DataType::StringValue(s) => Ok(embedding(s, &self.embeddings, &self.default_val)),
            DataType::StringArray(arr) => Ok(arr
                .values
                .iter()
                .flat_map(|s| embedding(s, &self.embeddings, &self.default_val))
                .collect()),
            _ => Err("Invalid data type for Embedding".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
        if self.embeddings.is_empty() {
            return Err(format!("Embedding for {}: embeddings must not be empty", name).into());
        }
        if self.default_val.is_empty() {
            return Err(format!("Embedding for {}: default_val must not be empty", name).into());
        }
        Ok(())
    }
}

/// Passes through numerical values with minimal transformation.
///
/// This transformation attempts to convert any input value to an f32,
/// preserving the original value with appropriate type casting.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Identity {
    /// Default value to use when transformation is not possible. Used when the feature value is None.
    pub default_val: Vec<f32>,
}

impl Transformation for Identity {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
        match feat_val {
            DataType::DoubleValue(val) => Ok(vec![*val as f32]),
            DataType::FloatValue(val) => Ok(vec![*val]),
            DataType::IntegerValue(val) => Ok(vec![*val as f32]),
            DataType::LongValue(val) => Ok(vec![*val as f32]),
            DataType::StringValue(s) => match s.parse::<f32>() {
                Ok(f) => Ok(vec![f]),
                Err(e) => Err(format!("Error in parsing StringValue.\n {}", e).into()),
            },
            DataType::DoubleArray(arr) => Ok(arr.values.iter().map(|v| *v as f32).collect()),
            DataType::FloatArray(arr) => Ok(arr.values.iter().map(|v| *v).collect()),
            DataType::IntegerArray(arr) => Ok(arr.values.iter().map(|v| *v as f32).collect()),
            DataType::LongArray(arr) => Ok(arr.values.iter().map(|v| *v as f32).collect()),
            DataType::StringArray(arr) => {
                let mut results = Vec::new();
                let mut errors = Vec::new();

                for s in &arr.values {
                    match s.parse::<f32>() {
                        Ok(f) => results.push(f),
                        Err(_) => errors.push(s.to_string()),
                    }
                }
                if errors.is_empty() {
                    Ok(results)
                } else {
                    Err(format!("Error in parsing StringArray.\n {}", errors.join(", ")).into())
                }
            }
        }
    }

    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
        if self.default_val.is_empty() {
            return Err(format!("Identity for {}: default_val must not be empty", name).into());
        }
        Ok(())
    }
}

/// Scales numerical values to the range [0, 1] using min-max normalization for 32-bit float values.
///
/// The formula used is: (value - min) / (max - min)
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct MinMaxScaling32 {
    /// Default value to use when transformation is not possible
    pub default_val: Vec<f32>,
    /// Minimum value in the original scale
    pub min: f32,
    /// Maximum value in the original scale
    pub max: f32,
}

impl Transformation for MinMaxScaling32 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
        match feat_val {
            DataType::FloatValue(val) => Ok(vec![min_max_scaling_32(*val, &self.min, &self.max)]),
            DataType::IntegerValue(val) => {
                Ok(vec![min_max_scaling_32(*val as f32, &self.min, &self.max)])
            }
            DataType::FloatArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_32(*v, &self.min, &self.max))
                .collect()),
            DataType::IntegerArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_32(*v as f32, &self.min, &self.max))
                .collect()),
            _ => Err("Invalid data type for MinMaxScaling32".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
        if self.min >= self.max {
            return Err(format!(
                "MinMaxScaling32 for {}: min ({}) must be less than max ({})",
                name, self.min, self.max
            )
            .into());
        }
        if self.default_val.is_empty() {
            return Err(format!(
                "MinMaxScaling32 for {}: default_val must not be empty",
                name
            )
            .into());
        }
        Ok(())
    }
}

/// Scales numerical values to the range [0, 1] using min-max normalization for 64-bit float values.
///
/// The formula used is: (value - min) / (max - min)
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct MinMaxScaling64 {
    /// Default value to use when transformation is not possible
    pub default_val: Vec<f32>,
    /// Minimum value in the original scale
    pub min: f64,
    /// Maximum value in the original scale
    pub max: f64,
}

impl Transformation for MinMaxScaling64 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
        match feat_val {
            DataType::DoubleValue(val) => {
                Ok(vec![min_max_scaling_64(*val, &self.min, &self.max) as f32])
            }
            DataType::LongValue(val) => {
                Ok(vec![
                    min_max_scaling_64(*val as f64, &self.min, &self.max) as f32
                ])
            }
            DataType::DoubleArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_64(*v, &self.min, &self.max))
                .map(|v| v as f32)
                .collect()),
            DataType::LongArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_64(*v as f64, &self.min, &self.max))
                .map(|v| v as f32)
                .collect()),
            _ => Err("Invalid data type for MinMaxScaling64".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
        if self.min >= self.max {
            return Err(format!(
                "MinMaxScaling64 for {}: min ({}) must be less than max ({})",
                name, self.min, self.max
            )
            .into());
        }
        if self.default_val.is_empty() {
            return Err(format!(
                "MinMaxScaling64 for {}: default_val must not be empty",
                name
            )
            .into());
        }
        Ok(())
    }
}

/// Converts categorical string values into one-hot encoded vectors.
///
/// For each input string, outputs a vector where all values are 0 except for the
/// position corresponding to the input category, which is set to 1.
/// categories are expected to be in a sorted order.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OneHotEncoding {
    /// Default value to use when a category is not found
    pub default_val: Vec<f32>,
    /// List of possible categories in sorted order
    pub categories: Vec<String>,
}

impl Transformation for OneHotEncoding {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
        match feat_val {
            DataType::StringValue(s) => {
                Ok(one_hot_encoding(s, &self.categories, &self.default_val))
            }
            DataType::StringArray(arr) => Ok(arr
                .values
                .iter()
                .flat_map(|s| one_hot_encoding(s, &self.categories, &self.default_val))
                .collect()),
            _ => Err("Invalid data type for OneHotEncoding".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
        if self.categories.is_empty() {
            return Err(
                format!("OneHotEncoding for {}: categories must not be empty", name).into(),
            );
        }
        if self.default_val.is_empty() {
            return Err(
                format!("OneHotEncoding for {}: default_val must not be empty", name).into(),
            );
        }
        Ok(())
    }
}

/// Standardizes numerical values to have zero mean and unit variance for 32-bit float values.
///
/// The formula used is: (value - mean) / std_dev
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Standardization32 {
    /// Default value to use when transformation is not possible
    pub default_val: Vec<f32>,
    /// Mean of the distribution
    pub mean: f32,
    /// Standard deviation of the distribution
    pub std_dev: f32,
}

impl Transformation for Standardization32 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
        match feat_val {
            DataType::FloatValue(val) => Ok(vec![standardize_32(*val, &self.mean, &self.std_dev)]),
            DataType::IntegerValue(val) => {
                Ok(vec![standardize_32(*val as f32, &self.mean, &self.std_dev)])
            }
            DataType::FloatArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_32(*v, &self.mean, &self.std_dev))
                .collect()),
            DataType::IntegerArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_32(*v as f32, &self.mean, &self.std_dev))
                .collect()),
            _ => Err("Invalid data type for Standardization32".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
        if self.std_dev <= 0.0 {
            return Err(format!(
                "Standardization32 for {}: std_dev ({}) must be greater than zero",
                name, self.std_dev
            )
            .into());
        }
        if self.default_val.is_empty() {
            return Err(format!(
                "Standardization32 for {}: default_val must not be empty",
                name
            )
            .into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Standardization64 {
    /// Default value to use when transformation is not possible
    pub default_val: Vec<f32>,
    /// Mean of the distribution
    pub mean: f64,
    /// Standard deviation of the distribution
    pub std_dev: f64,
}

impl Transformation for Standardization64 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>> {
        match feat_val {
            DataType::DoubleValue(val) => {
                Ok(vec![standardize_64(*val, &self.mean, &self.std_dev) as f32])
            }
            DataType::LongValue(val) => {
                Ok(vec![
                    standardize_64(*val as f64, &self.mean, &self.std_dev) as f32
                ])
            }
            DataType::DoubleArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_64(*v, &self.mean, &self.std_dev))
                .map(|v| v as f32)
                .collect()),
            DataType::LongArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_64(*v as f64, &self.mean, &self.std_dev))
                .map(|v| v as f32)
                .collect()),
            _ => Err("Invalid data type for Standardization64".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), Box<dyn Error>> {
        if self.std_dev <= 0.0 {
            return Err(format!(
                "Standardization64 for {}: std_dev ({}) must be greater than zero",
                name, self.std_dev
            )
            .into());
        }
        if self.default_val.is_empty() {
            return Err(format!(
                "Standardization64 for {}: default_val must not be empty",
                name
            )
            .into());
        }
        Ok(())
    }
}

fn embedding(s: &String, embeddings: &HashMap<String, Vec<f32>>, default_val: &[f32]) -> Vec<f32> {
    if let Some(embedding) = embeddings.get(s) {
        embedding.clone()
    } else {
        default_val.to_vec()
    }
}

fn min_max_scaling_32(val: f32, min: &f32, max: &f32) -> f32 {
    (val - min) / (max - min)
}

fn min_max_scaling_64(val: f64, min: &f64, max: &f64) -> f64 {
    (val - min) / (max - min)
}

fn one_hot_encoding(s: &String, categories: &Vec<String>, default_val: &[f32]) -> Vec<f32> {
    let mut one_hot = vec![0.0; categories.len()];
    if let Ok(index) = categories.binary_search(s) {
        one_hot[index] = 1.0;
        one_hot
    } else {
        default_val.to_vec()
    }
}

fn standardize_32(val: f32, mean: &f32, std_dev: &f32) -> f32 {
    (val - mean) / std_dev
}

fn standardize_64(val: f64, mean: &f64, std_dev: &f64) -> f64 {
    (val - mean) / std_dev
}

#[cfg(test)]
mod tests {
    use super::*;
    use hushar_proto::hushar::data_type::DataType;
    use std::collections::HashMap;

    fn assert_float_eq(a: f32, b: f32) {
        const EPSILON: f32 = 1e-8;
        if a.is_nan() {
            assert!(b.is_nan());
        } else if a.is_infinite() {
            assert!(b.is_infinite());
            assert_eq!(a.is_sign_positive(), b.is_sign_positive());
        } else {
            assert!(
                (a - b).abs() < EPSILON,
                "Expected {} to be close to {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_embedding_transform_string_value() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);
        embeddings.insert("dog".to_string(), vec![0.4, 0.5, 0.6]);

        let default_val = vec![0.0, 0.0, 0.0];
        let embedding = Embedding {
            embeddings,
            default_val: default_val.clone(),
        };

        // Test existing embedding
        let result = embedding
            .transform(&DataType::StringValue("cat".to_string()))
            .unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        // Test non-existent embedding (should return default)
        let result = embedding
            .transform(&DataType::StringValue("bird".to_string()))
            .unwrap();
        assert_eq!(result, default_val);
    }

    #[test]
    fn test_embedding_transform_string_array() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2]);
        embeddings.insert("dog".to_string(), vec![0.3, 0.4]);

        let default_val = vec![0.0, 0.0];
        let embedding = Embedding {
            embeddings,
            default_val: default_val.clone(),
        };

        let string_array = hushar_proto::hushar::StringArray {
            values: vec!["cat".to_string(), "bird".to_string(), "dog".to_string()],
        };

        let result = embedding
            .transform(&DataType::StringArray(string_array))
            .unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.0, 0.0, 0.3, 0.4]);
    }

    #[test]
    fn test_embedding_validate() {
        // Valid embedding
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);

        let embedding = Embedding {
            embeddings,
            default_val: vec![0.0, 0.0, 0.0],
        };

        assert!(embedding.validate("test_feature").is_ok());

        // Invalid: empty embeddings
        let embedding = Embedding {
            embeddings: HashMap::new(),
            default_val: vec![0.0, 0.0, 0.0],
        };

        assert!(embedding.validate("test_feature").is_err());

        // Invalid: empty default_val
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);

        let embedding = Embedding {
            embeddings,
            default_val: vec![],
        };

        assert!(embedding.validate("test_feature").is_err());
    }

    #[test]
    fn test_identity_validate() {
        // Valid
        let identity = Identity {
            default_val: vec![1.0],
        };

        assert!(identity.validate("test_feature").is_ok());

        // Invalid: empty default_val
        let identity = Identity {
            default_val: vec![],
        };

        assert!(identity.validate("test_feature").is_err());
    }

    #[test]
    fn test_transform_double_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::DoubleValue(42.5);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_float_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::FloatValue(42.5);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_integer_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::IntegerValue(42);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.0_f32);
    }

    #[test]
    fn test_transform_long_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::LongValue(42_i64);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.0_f32);
    }

    #[test]
    fn test_transform_string_value_valid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::StringValue("42.5".to_string());
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_string_value_invalid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::StringValue("not_a_number".to_string());
        let result = identity.transform(&input);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Error in parsing StringValue"));
    }

    #[test]
    fn test_transform_double_array() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::DoubleArray(hushar_proto::hushar::DoubleArray {
            values: vec![1.1, 2.2, 3.3],
        });
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 3);
        assert_float_eq(result[0], 1.1_f32);
        assert_float_eq(result[1], 2.2_f32);
        assert_float_eq(result[2], 3.3_f32);
    }

    #[test]
    fn test_transform_float_array() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::FloatArray(hushar_proto::hushar::FloatArray {
            values: vec![1.1, 2.2, 3.3],
        });
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 3);
        assert_float_eq(result[0], 1.1_f32);
        assert_float_eq(result[1], 2.2_f32);
        assert_float_eq(result[2], 3.3_f32);
    }

    #[test]
    fn test_transform_integer_array() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::IntegerArray(hushar_proto::hushar::IntegerArray {
            values: vec![1, 2, 3],
        });
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 3);
        assert_float_eq(result[0], 1.0_f32);
        assert_float_eq(result[1], 2.0_f32);
        assert_float_eq(result[2], 3.0_f32);
    }

    #[test]
    fn test_transform_long_array() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::LongArray(hushar_proto::hushar::LongArray {
            values: vec![1, 2, 3],
        });
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 3);
        assert_float_eq(result[0], 1.0_f32);
        assert_float_eq(result[1], 2.0_f32);
        assert_float_eq(result[2], 3.0_f32);
    }

    #[test]
    fn test_transform_string_array_all_valid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::StringArray(hushar_proto::hushar::StringArray {
            values: vec!["1.1".to_string(), "2.2".to_string(), "3.3".to_string()],
        });
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 3);
        assert_float_eq(result[0], 1.1_f32);
        assert_float_eq(result[1], 2.2_f32);
        assert_float_eq(result[2], 3.3_f32);
    }

    #[test]
    fn test_transform_string_array_with_invalid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::StringArray(hushar_proto::hushar::StringArray {
            values: vec![
                "1.1".to_string(),
                "not_a_number".to_string(),
                "3.3".to_string(),
            ],
        });
        let result = identity.transform(&input);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Error in parsing StringArray"));
    }

    #[test]
    fn test_transform_string_array_empty() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = DataType::StringArray(hushar_proto::hushar::StringArray { values: vec![] });
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_min_max_scaling32_transform_float() {
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        // Test single value
        let result = scaler.transform(&DataType::FloatValue(5.0)).unwrap();
        assert_eq!(result, vec![0.5]);

        // Test out of range values
        let result = scaler.transform(&DataType::FloatValue(-5.0)).unwrap();
        assert_eq!(result, vec![-0.5]);

        let result = scaler.transform(&DataType::FloatValue(15.0)).unwrap();
        assert_eq!(result, vec![1.5]);

        // Test integer value
        let result = scaler.transform(&DataType::IntegerValue(5)).unwrap();
        assert_eq!(result, vec![0.5]);
    }

    #[test]
    fn test_min_max_scaling32_transform_array() {
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        // Test float array
        let float_array = hushar_proto::hushar::FloatArray {
            values: vec![0.0, 5.0, 10.0],
        };

        let result = scaler
            .transform(&DataType::FloatArray(float_array))
            .unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);

        // Test integer array
        let int_array = hushar_proto::hushar::IntegerArray {
            values: vec![0, 5, 10],
        };

        let result = scaler
            .transform(&DataType::IntegerArray(int_array))
            .unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_min_max_scaling32_validate() {
        // Valid
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_ok());

        // Invalid: min >= max
        let scaler = MinMaxScaling32 {
            min: 10.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_err());

        // Invalid: empty default_val
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![],
        };

        assert!(scaler.validate("test_feature").is_err());
    }

    #[test]
    fn test_min_max_scaling64_transform_double() {
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        // Test single value
        let result = scaler.transform(&DataType::DoubleValue(5.0)).unwrap();
        assert_eq!(result, vec![0.5]);

        // Test out of range values
        let result = scaler.transform(&DataType::DoubleValue(-5.0)).unwrap();
        assert_eq!(result, vec![-0.5]);

        let result = scaler.transform(&DataType::DoubleValue(15.0)).unwrap();
        assert_eq!(result, vec![1.5]);

        // Test long value
        let result = scaler.transform(&DataType::LongValue(5)).unwrap();
        assert_eq!(result, vec![0.5]);
    }

    #[test]
    fn test_min_max_scaling64_transform_array() {
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        // Test double array
        let double_array = hushar_proto::hushar::DoubleArray {
            values: vec![0.0, 5.0, 10.0],
        };

        let result = scaler
            .transform(&DataType::DoubleArray(double_array))
            .unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);

        // Test long array
        let long_array = hushar_proto::hushar::LongArray {
            values: vec![0, 5, 10],
        };

        let result = scaler.transform(&DataType::LongArray(long_array)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_min_max_scaling64_validate() {
        // Valid
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_ok());

        // Invalid: min >= max
        let scaler = MinMaxScaling64 {
            min: 10.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_err());

        // Invalid: empty default_val
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![],
        };

        assert!(scaler.validate("test_feature").is_err());
    }

    #[test]
    fn test_one_hot_encoding_transform_string_value() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        let one_hot = OneHotEncoding {
            categories,
            default_val: default_val.clone(),
        };

        // Test existing category
        let result = one_hot
            .transform(&DataType::StringValue("cat".to_string()))
            .unwrap();
        assert_eq!(result, vec![1.0, 0.0, 0.0]);

        let result = one_hot
            .transform(&DataType::StringValue("dog".to_string()))
            .unwrap();
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        // Test non-existent category (should return default)
        let result = one_hot
            .transform(&DataType::StringValue("bird".to_string()))
            .unwrap();
        assert_eq!(result, default_val);
    }

    #[test]
    fn test_one_hot_encoding_transform_string_array() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        let one_hot = OneHotEncoding {
            categories,
            default_val: default_val.clone(),
        };

        let string_array = hushar_proto::hushar::StringArray {
            values: vec!["cat".to_string(), "bird".to_string(), "fish".to_string()],
        };

        let result = one_hot
            .transform(&DataType::StringArray(string_array))
            .unwrap();
        // Expected: cat one-hot + bird default + fish one-hot
        assert_eq!(result, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_one_hot_encoding_validate() {
        // Valid
        let one_hot = OneHotEncoding {
            categories: vec!["cat".to_string(), "dog".to_string()],
            default_val: vec![0.0, 0.0],
        };

        assert!(one_hot.validate("test_feature").is_ok());

        // Invalid: empty categories
        let one_hot = OneHotEncoding {
            categories: vec![],
            default_val: vec![0.0, 0.0],
        };

        assert!(one_hot.validate("test_feature").is_err());

        // Invalid: empty default_val
        let one_hot = OneHotEncoding {
            categories: vec!["cat".to_string(), "dog".to_string()],
            default_val: vec![],
        };

        assert!(one_hot.validate("test_feature").is_err());
    }

    #[test]
    fn test_standardization32_transform_float() {
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        // Test standardization
        let result = standardizer.transform(&DataType::FloatValue(60.0)).unwrap();
        assert_eq!(result, vec![1.0]);

        let result = standardizer.transform(&DataType::FloatValue(40.0)).unwrap();
        assert_eq!(result, vec![-1.0]);

        let result = standardizer.transform(&DataType::FloatValue(50.0)).unwrap();
        assert_eq!(result, vec![0.0]);

        // Test integer value
        let result = standardizer.transform(&DataType::IntegerValue(60)).unwrap();
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_standardization32_transform_array() {
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        // Test float array
        let float_array = hushar_proto::hushar::FloatArray {
            values: vec![40.0, 50.0, 60.0],
        };

        let result = standardizer
            .transform(&DataType::FloatArray(float_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);

        // Test integer array
        let int_array = hushar_proto::hushar::IntegerArray {
            values: vec![40, 50, 60],
        };

        let result = standardizer
            .transform(&DataType::IntegerArray(int_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_standardization32_validate() {
        // Valid
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_ok());

        // Invalid: std_dev <= 0
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 0.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_err());

        // Invalid: empty default_val
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![],
        };

        assert!(standardizer.validate("test_feature").is_err());
    }

    #[test]
    fn test_standardization64_transform_double() {
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        // Test standardization
        let result = standardizer
            .transform(&DataType::DoubleValue(60.0))
            .unwrap();
        assert_eq!(result, vec![1.0]);

        let result = standardizer
            .transform(&DataType::DoubleValue(40.0))
            .unwrap();
        assert_eq!(result, vec![-1.0]);

        let result = standardizer
            .transform(&DataType::DoubleValue(50.0))
            .unwrap();
        assert_eq!(result, vec![0.0]);

        // Test long value
        let result = standardizer.transform(&DataType::LongValue(60)).unwrap();
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_standardization64_transform_array() {
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        // Test double array
        let double_array = hushar_proto::hushar::DoubleArray {
            values: vec![40.0, 50.0, 60.0],
        };

        let result = standardizer
            .transform(&DataType::DoubleArray(double_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);

        // Test long array
        let long_array = hushar_proto::hushar::LongArray {
            values: vec![40, 50, 60],
        };

        let result = standardizer
            .transform(&DataType::LongArray(long_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_standardization64_validate() {
        // Valid
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_ok());

        // Invalid: std_dev <= 0
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 0.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_err());

        // Invalid: empty default_val
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![],
        };

        assert!(standardizer.validate("test_feature").is_err());
    }

    #[test]
    fn test_transform_invalid_data_types() {
        // Test cases for invalid data types for each transformation

        // Embedding with non-string types
        let embedding = Embedding {
            embeddings: HashMap::new(),
            default_val: vec![0.0],
        };

        assert!(embedding.transform(&DataType::FloatValue(1.0)).is_err());

        // MinMaxScaling32 with string type
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler
            .transform(&DataType::StringValue("test".to_string()))
            .is_err());

        // MinMaxScaling64 with string type
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler
            .transform(&DataType::StringValue("test".to_string()))
            .is_err());

        // OneHotEncoding with numeric types
        let one_hot = OneHotEncoding {
            categories: vec!["cat".to_string()],
            default_val: vec![0.0],
        };

        assert!(one_hot.transform(&DataType::FloatValue(1.0)).is_err());

        // Standardization32 with string type
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(standardizer
            .transform(&DataType::StringValue("test".to_string()))
            .is_err());

        // Standardization64 with string type
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(standardizer
            .transform(&DataType::StringValue("test".to_string()))
            .is_err());
    }

    #[test]
    fn test_embedding_helper() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);

        let default_val = vec![0.0, 0.0, 0.0];

        // Test existing embedding
        let result = embedding(&"cat".to_string(), &embeddings, &default_val);
        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        // Test non-existent embedding
        let result = embedding(&"dog".to_string(), &embeddings, &default_val);
        assert_eq!(result, default_val);
    }

    #[test]
    fn test_min_max_scaling_32_helper() {
        let min = 0.0;
        let max = 10.0;

        // Test within range
        let result = min_max_scaling_32(5.0, &min, &max);
        assert_eq!(result, 0.5);

        // Test at boundaries
        let result = min_max_scaling_32(0.0, &min, &max);
        assert_eq!(result, 0.0);

        let result = min_max_scaling_32(10.0, &min, &max);
        assert_eq!(result, 1.0);

        // Test outside range
        let result = min_max_scaling_32(-5.0, &min, &max);
        assert_eq!(result, -0.5);

        let result = min_max_scaling_32(15.0, &min, &max);
        assert_eq!(result, 1.5);
    }

    #[test]
    fn test_min_max_scaling_64_helper() {
        let min = 0.0;
        let max = 10.0;

        // Test within range
        let result = min_max_scaling_64(5.0, &min, &max);
        assert_eq!(result, 0.5);

        // Test at boundaries
        let result = min_max_scaling_64(0.0, &min, &max);
        assert_eq!(result, 0.0);

        let result = min_max_scaling_64(10.0, &min, &max);
        assert_eq!(result, 1.0);

        // Test outside range
        let result = min_max_scaling_64(-5.0, &min, &max);
        assert_eq!(result, -0.5);

        let result = min_max_scaling_64(15.0, &min, &max);
        assert_eq!(result, 1.5);
    }

    #[test]
    fn test_one_hot_encoding_helper() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        // Test existing categories
        let result = one_hot_encoding(&"cat".to_string(), &categories, &default_val);
        assert_eq!(result, vec![1.0, 0.0, 0.0]);

        let result = one_hot_encoding(&"dog".to_string(), &categories, &default_val);
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        let result = one_hot_encoding(&"fish".to_string(), &categories, &default_val);
        assert_eq!(result, vec![0.0, 0.0, 1.0]);

        // Test non-existent category
        let result = one_hot_encoding(&"bird".to_string(), &categories, &default_val);
        assert_eq!(result, default_val);
    }

    #[test]
    fn test_standardize_32_helper() {
        let mean = 50.0;
        let std_dev = 10.0;

        // Test standardization
        let result = standardize_32(40.0, &mean, &std_dev);
        assert_eq!(result, -1.0);

        let result = standardize_32(50.0, &mean, &std_dev);
        assert_eq!(result, 0.0);

        let result = standardize_32(60.0, &mean, &std_dev);
        assert_eq!(result, 1.0);

        let result = standardize_32(70.0, &mean, &std_dev);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_standardize_64_helper() {
        let mean = 50.0;
        let std_dev = 10.0;

        // Test standardization
        let result = standardize_64(40.0, &mean, &std_dev);
        assert_eq!(result, -1.0);

        let result = standardize_64(50.0, &mean, &std_dev);
        assert_eq!(result, 0.0);

        let result = standardize_64(60.0, &mean, &std_dev);
        assert_eq!(result, 1.0);

        let result = standardize_64(70.0, &mean, &std_dev);
        assert_eq!(result, 2.0);
    }
}
