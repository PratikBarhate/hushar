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

use hushar::hushar_proto::data_type::DataType as Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Turns one feature value into the numbers a model input takes.
pub trait Transformation: std::fmt::Debug + Send + Sync {
    /// Returns the default value to use when transformation is not possible.
    fn get_default_val(&self) -> &[f32];

    /// Transforms one feature value into the values it contributes to a row.
    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError>;

    /// The same transformation, in double precision.
    ///
    /// The default widens the `f32` result, which is exact but no more precise than what
    /// it started from. The 64-bit scalers and [`Identity`] override it, so a model
    /// configured for doubles gets its arithmetic done in doubles. That matters for tree
    /// ensembles: a regression tree is not continuous, so a value nudged across a split
    /// by a narrowing cast selects a different leaf rather than shifting slightly.
    fn transform_f64(
        &self,
        feat_val: &Value,
    ) -> Result<Vec<f64>, crate::inference::InferenceError> {
        Ok(self
            .transform(feat_val)?
            .into_iter()
            .map(f64::from)
            .collect())
    }

    /// The default value in double precision, for an absent feature.
    fn default_val_f64(&self) -> Vec<f64> {
        self.get_default_val()
            .iter()
            .copied()
            .map(f64::from)
            .collect()
    }

    /// Checks the transformation's own parameters, naming `name` in any error.
    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError>;
}

/// Maps string values to vector embeddings using a pre-defined embedding table.
///
/// If the string value is not found in the embedding table, the default value is used.
///
/// Fields:
/// - `default_val` — used when a string is not in the table.
/// - `embeddings` — string value to its vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Embedding {
    pub default_val: Vec<f32>,
    pub embeddings: HashMap<String, Vec<f32>>,
}

impl Transformation for Embedding {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError> {
        match feat_val {
            Value::StringValue(s) => Ok(embedding(s, &self.embeddings, &self.default_val)),
            Value::StringArray(arr) => Ok(arr
                .values
                .iter()
                .flat_map(|s| embedding(s, &self.embeddings, &self.default_val))
                .collect()),
            _ => Err("Invalid data type for Embedding".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
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
/// This transformation attempts to convert any input value to f32,
/// preserving the original value with appropriate type casting.
///
/// Fields:
/// - `default_val` — used when the feature value is absent.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Identity {
    pub default_val: Vec<f32>,
}

impl Transformation for Identity {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }
    /// A boolean feature is the one-or-zero indicator it already is, which is what
    /// every model expects of a flag.
    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError> {
        match feat_val {
            Value::DoubleValue(val) => Ok(vec![*val as f32]),
            Value::FloatValue(val) => Ok(vec![*val]),
            Value::IntegerValue(val) => Ok(vec![*val as f32]),
            Value::LongValue(val) => Ok(vec![*val as f32]),
            Value::BoolValue(val) => Ok(vec![if *val { 1.0 } else { 0.0 }]),
            Value::StringValue(s) => match s.parse::<f32>() {
                Ok(f) => Ok(vec![f]),
                Err(e) => Err(format!("Error in parsing StringValue.\n {}", e).into()),
            },
            Value::DoubleArray(arr) => Ok(arr.values.iter().map(|v| *v as f32).collect()),
            Value::FloatArray(arr) => Ok(arr.values.to_vec()),
            Value::IntegerArray(arr) => Ok(arr.values.iter().map(|v| *v as f32).collect()),
            Value::LongArray(arr) => Ok(arr.values.iter().map(|v| *v as f32).collect()),
            Value::StringArray(arr) => {
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

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
        if self.default_val.is_empty() {
            return Err(format!("Identity for {}: default_val must not be empty", name).into());
        }
        Ok(())
    }

    /// Passes the value through without ever narrowing it.
    ///
    /// The whole point of `Identity` is that the model sees what the caller sent, so
    /// a `double` feature reaching a `double` input must not lose its low bits by
    /// travelling through an `f32` on the way.
    fn transform_f64(
        &self,
        feat_val: &Value,
    ) -> Result<Vec<f64>, crate::inference::InferenceError> {
        match feat_val {
            Value::DoubleValue(val) => Ok(vec![*val]),
            Value::FloatValue(val) => Ok(vec![f64::from(*val)]),
            Value::IntegerValue(val) => Ok(vec![f64::from(*val)]),
            Value::LongValue(val) => Ok(vec![*val as f64]),
            Value::BoolValue(val) => Ok(vec![if *val { 1.0 } else { 0.0 }]),
            Value::StringValue(s) => match s.parse::<f64>() {
                Ok(f) => Ok(vec![f]),
                Err(e) => Err(format!("Error in parsing StringValue.\n {}", e).into()),
            },
            Value::DoubleArray(arr) => Ok(arr.values.clone()),
            Value::FloatArray(arr) => Ok(arr.values.iter().copied().map(f64::from).collect()),
            Value::IntegerArray(arr) => Ok(arr.values.iter().copied().map(f64::from).collect()),
            Value::LongArray(arr) => Ok(arr.values.iter().map(|v| *v as f64).collect()),
            Value::StringArray(arr) => {
                let mut results = Vec::new();
                let mut errors = Vec::new();
                for s in &arr.values {
                    match s.parse::<f64>() {
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
}

/// Scales a 32-bit value into `[0, 1]` as `(value - min) / (max - min)`.
///
/// Fields:
/// - `default_val` — used when the value cannot be scaled.
/// - `min`, `max` — the original scale.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct MinMaxScaling32 {
    pub default_val: Vec<f32>,
    pub min: f32,
    pub max: f32,
}

impl Transformation for MinMaxScaling32 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError> {
        match feat_val {
            Value::FloatValue(val) => Ok(vec![min_max_scaling_32(*val, &self.min, &self.max)]),
            Value::IntegerValue(val) => {
                Ok(vec![min_max_scaling_32(*val as f32, &self.min, &self.max)])
            }
            Value::FloatArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_32(*v, &self.min, &self.max))
                .collect()),
            Value::IntegerArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_32(*v as f32, &self.min, &self.max))
                .collect()),
            _ => Err("Invalid data type for MinMaxScaling32".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
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

/// Scales a 64-bit value into `[0, 1]` as `(value - min) / (max - min)`.
///
/// Fields:
/// - `default_val` — used when the value cannot be scaled.
/// - `min`, `max` — the original scale.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct MinMaxScaling64 {
    pub default_val: Vec<f32>,
    pub min: f64,
    pub max: f64,
}

impl Transformation for MinMaxScaling64 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError> {
        match feat_val {
            Value::DoubleValue(val) => {
                Ok(vec![min_max_scaling_64(*val, &self.min, &self.max) as f32])
            }
            Value::LongValue(val) => Ok(vec![
                min_max_scaling_64(*val as f64, &self.min, &self.max) as f32,
            ]),
            Value::DoubleArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_64(*v, &self.min, &self.max))
                .map(|v| v as f32)
                .collect()),
            Value::LongArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_64(*v as f64, &self.min, &self.max))
                .map(|v| v as f32)
                .collect()),
            _ => Err("Invalid data type for MinMaxScaling64".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
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

    /// The same scaling, kept in double precision.
    ///
    /// [`MinMaxScaling64::transform`] already computes in `f64` and then narrows for
    /// the vectorised path. This is that arithmetic without the last step, which is
    /// the only reason to configure a 64-bit scaler against a 64-bit model.
    fn transform_f64(
        &self,
        feat_val: &Value,
    ) -> Result<Vec<f64>, crate::inference::InferenceError> {
        match feat_val {
            Value::DoubleValue(val) => Ok(vec![min_max_scaling_64(*val, &self.min, &self.max)]),
            Value::LongValue(val) => {
                Ok(vec![min_max_scaling_64(*val as f64, &self.min, &self.max)])
            }
            Value::DoubleArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_64(*v, &self.min, &self.max))
                .collect()),
            Value::LongArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| min_max_scaling_64(*v as f64, &self.min, &self.max))
                .collect()),
            _ => Err("Invalid data type for MinMaxScaling64".into()),
        }
    }
}

/// Converts categorical string values into one-hot encoded vectors.
///
/// For each input string, outputs a vector where all values are 0 except for the
/// position corresponding to the input category, which is set to 1.
/// categories are expected to be in a sorted order.
///
/// Fields:
/// - `default_val` — used for an unknown category.
/// - `categories` — the categories, which must be sorted.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OneHotEncoding {
    pub default_val: Vec<f32>,
    pub categories: Vec<String>,
}

impl Transformation for OneHotEncoding {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError> {
        match feat_val {
            Value::StringValue(s) => Ok(one_hot_encoding(s, &self.categories, &self.default_val)),
            Value::StringArray(arr) => Ok(arr
                .values
                .iter()
                .flat_map(|s| one_hot_encoding(s, &self.categories, &self.default_val))
                .collect()),
            _ => Err("Invalid data type for OneHotEncoding".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
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
///
/// Fields:
/// - `default_val` — used when the value cannot be standardized.
/// - `mean`, `std_dev` — the distribution.
///
/// Fields:
/// - `default_val` — used when the value cannot be standardized.
/// - `mean`, `std_dev` — the distribution.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Standardization32 {
    pub default_val: Vec<f32>,
    pub mean: f32,
    pub std_dev: f32,
}

impl Transformation for Standardization32 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError> {
        match feat_val {
            Value::FloatValue(val) => Ok(vec![standardize_32(*val, &self.mean, &self.std_dev)]),
            Value::IntegerValue(val) => {
                Ok(vec![standardize_32(*val as f32, &self.mean, &self.std_dev)])
            }
            Value::FloatArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_32(*v, &self.mean, &self.std_dev))
                .collect()),
            Value::IntegerArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_32(*v as f32, &self.mean, &self.std_dev))
                .collect()),
            _ => Err("Invalid data type for Standardization32".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
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
    pub default_val: Vec<f32>,
    pub mean: f64,
    pub std_dev: f64,
}

impl Transformation for Standardization64 {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    fn transform(&self, feat_val: &Value) -> Result<Vec<f32>, crate::inference::InferenceError> {
        match feat_val {
            Value::DoubleValue(val) => {
                Ok(vec![standardize_64(*val, &self.mean, &self.std_dev) as f32])
            }
            Value::LongValue(val) => Ok(vec![
                standardize_64(*val as f64, &self.mean, &self.std_dev) as f32,
            ]),
            Value::DoubleArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_64(*v, &self.mean, &self.std_dev))
                .map(|v| v as f32)
                .collect()),
            Value::LongArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_64(*v as f64, &self.mean, &self.std_dev))
                .map(|v| v as f32)
                .collect()),
            _ => Err("Invalid data type for Standardization64".into()),
        }
    }

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
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

    /// The same standardization, kept in double precision.
    fn transform_f64(
        &self,
        feat_val: &Value,
    ) -> Result<Vec<f64>, crate::inference::InferenceError> {
        match feat_val {
            Value::DoubleValue(val) => Ok(vec![standardize_64(*val, &self.mean, &self.std_dev)]),
            Value::LongValue(val) => {
                Ok(vec![standardize_64(*val as f64, &self.mean, &self.std_dev)])
            }
            Value::DoubleArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_64(*v, &self.mean, &self.std_dev))
                .collect()),
            Value::LongArray(arr) => Ok(arr
                .values
                .iter()
                .map(|v| standardize_64(*v as f64, &self.mean, &self.std_dev))
                .collect()),
            _ => Err("Invalid data type for Standardization64".into()),
        }
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

fn one_hot_encoding(s: &String, categories: &[String], default_val: &[f32]) -> Vec<f32> {
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
    use hushar::hushar_proto::data_type::DataType as Value;
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

    /// Test existing embedding Test non-existent embedding (should return default)
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

        let result = embedding
            .transform(&Value::StringValue("cat".to_string()))
            .unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        let result = embedding
            .transform(&Value::StringValue("bird".to_string()))
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

        let string_array = hushar::hushar_proto::StringArray {
            values: vec!["cat".to_string(), "bird".to_string(), "dog".to_string()],
        };

        let result = embedding
            .transform(&Value::StringArray(string_array))
            .unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.0, 0.0, 0.3, 0.4]);
    }

    /// Valid embedding Invalid: empty embeddings Invalid: empty default_val
    #[test]
    fn test_embedding_validate() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);

        let embedding = Embedding {
            embeddings,
            default_val: vec![0.0, 0.0, 0.0],
        };

        assert!(embedding.validate("test_feature").is_ok());

        let embedding = Embedding {
            embeddings: HashMap::new(),
            default_val: vec![0.0, 0.0, 0.0],
        };

        assert!(embedding.validate("test_feature").is_err());

        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);

        let embedding = Embedding {
            embeddings,
            default_val: vec![],
        };

        assert!(embedding.validate("test_feature").is_err());
    }

    /// Valid Invalid: empty default_val
    #[test]
    fn test_identity_validate() {
        let identity = Identity {
            default_val: vec![1.0],
        };

        assert!(identity.validate("test_feature").is_ok());

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
        let input = Value::DoubleValue(42.5);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_float_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = Value::FloatValue(42.5);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_integer_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = Value::IntegerValue(42);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.0_f32);
    }

    #[test]
    fn test_transform_long_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = Value::LongValue(42_i64);
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.0_f32);
    }

    #[test]
    fn test_transform_string_value_valid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = Value::StringValue("42.5".to_string());
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_string_value_invalid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = Value::StringValue("not_a_number".to_string());
        let result = identity.transform(&input);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Error in parsing StringValue")
        );
    }

    #[test]
    fn test_transform_double_array() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = Value::DoubleArray(hushar::hushar_proto::DoubleArray {
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
        let input = Value::FloatArray(hushar::hushar_proto::FloatArray {
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
        let input = Value::IntegerArray(hushar::hushar_proto::IntegerArray {
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
        let input = Value::LongArray(hushar::hushar_proto::LongArray {
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
        let input = Value::StringArray(hushar::hushar_proto::StringArray {
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
        let input = Value::StringArray(hushar::hushar_proto::StringArray {
            values: vec![
                "1.1".to_string(),
                "not_a_number".to_string(),
                "3.3".to_string(),
            ],
        });
        let result = identity.transform(&input);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Error in parsing StringArray")
        );
    }

    #[test]
    fn test_transform_string_array_empty() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = Value::StringArray(hushar::hushar_proto::StringArray { values: vec![] });
        let result = identity.transform(&input).unwrap();
        assert_eq!(result.len(), 0);
    }

    /// Test single value Test out of range values Test integer value
    #[test]
    fn test_min_max_scaling32_transform_float() {
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let result = scaler.transform(&Value::FloatValue(5.0)).unwrap();
        assert_eq!(result, vec![0.5]);

        let result = scaler.transform(&Value::FloatValue(-5.0)).unwrap();
        assert_eq!(result, vec![-0.5]);

        let result = scaler.transform(&Value::FloatValue(15.0)).unwrap();
        assert_eq!(result, vec![1.5]);

        let result = scaler.transform(&Value::IntegerValue(5)).unwrap();
        assert_eq!(result, vec![0.5]);
    }

    /// Test float array Test integer array
    #[test]
    fn test_min_max_scaling32_transform_array() {
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let float_array = hushar::hushar_proto::FloatArray {
            values: vec![0.0, 5.0, 10.0],
        };

        let result = scaler.transform(&Value::FloatArray(float_array)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);

        let int_array = hushar::hushar_proto::IntegerArray {
            values: vec![0, 5, 10],
        };

        let result = scaler.transform(&Value::IntegerArray(int_array)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);
    }

    /// Valid Invalid: min >= max Invalid: empty default_val
    #[test]
    fn test_min_max_scaling32_validate() {
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_ok());

        let scaler = MinMaxScaling32 {
            min: 10.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_err());

        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![],
        };

        assert!(scaler.validate("test_feature").is_err());
    }

    /// Test single value Test out of range values Test long value
    #[test]
    fn test_min_max_scaling64_transform_double() {
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let result = scaler.transform(&Value::DoubleValue(5.0)).unwrap();
        assert_eq!(result, vec![0.5]);

        let result = scaler.transform(&Value::DoubleValue(-5.0)).unwrap();
        assert_eq!(result, vec![-0.5]);

        let result = scaler.transform(&Value::DoubleValue(15.0)).unwrap();
        assert_eq!(result, vec![1.5]);

        let result = scaler.transform(&Value::LongValue(5)).unwrap();
        assert_eq!(result, vec![0.5]);
    }

    /// Test double array Test long array
    #[test]
    fn test_min_max_scaling64_transform_array() {
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let double_array = hushar::hushar_proto::DoubleArray {
            values: vec![0.0, 5.0, 10.0],
        };

        let result = scaler.transform(&Value::DoubleArray(double_array)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);

        let long_array = hushar::hushar_proto::LongArray {
            values: vec![0, 5, 10],
        };

        let result = scaler.transform(&Value::LongArray(long_array)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);
    }

    /// Valid Invalid: min >= max Invalid: empty default_val
    #[test]
    fn test_min_max_scaling64_validate() {
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_ok());

        let scaler = MinMaxScaling64 {
            min: 10.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.validate("test_feature").is_err());

        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![],
        };

        assert!(scaler.validate("test_feature").is_err());
    }

    /// Test existing category Test non-existent category (should return default)
    #[test]
    fn test_one_hot_encoding_transform_string_value() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        let one_hot = OneHotEncoding {
            categories,
            default_val: default_val.clone(),
        };

        let result = one_hot
            .transform(&Value::StringValue("cat".to_string()))
            .unwrap();
        assert_eq!(result, vec![1.0, 0.0, 0.0]);

        let result = one_hot
            .transform(&Value::StringValue("dog".to_string()))
            .unwrap();
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        let result = one_hot
            .transform(&Value::StringValue("bird".to_string()))
            .unwrap();
        assert_eq!(result, default_val);
    }

    /// Expected: cat one-hot + bird default + fish one-hot
    #[test]
    fn test_one_hot_encoding_transform_string_array() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        let one_hot = OneHotEncoding {
            categories,
            default_val: default_val.clone(),
        };

        let string_array = hushar::hushar_proto::StringArray {
            values: vec!["cat".to_string(), "bird".to_string(), "fish".to_string()],
        };

        let result = one_hot
            .transform(&Value::StringArray(string_array))
            .unwrap();
        assert_eq!(result, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// Valid Invalid: empty categories Invalid: empty default_val
    #[test]
    fn test_one_hot_encoding_validate() {
        let one_hot = OneHotEncoding {
            categories: vec!["cat".to_string(), "dog".to_string()],
            default_val: vec![0.0, 0.0],
        };

        assert!(one_hot.validate("test_feature").is_ok());

        let one_hot = OneHotEncoding {
            categories: vec![],
            default_val: vec![0.0, 0.0],
        };

        assert!(one_hot.validate("test_feature").is_err());

        let one_hot = OneHotEncoding {
            categories: vec!["cat".to_string(), "dog".to_string()],
            default_val: vec![],
        };

        assert!(one_hot.validate("test_feature").is_err());
    }

    /// Test standardization Test integer value
    #[test]
    fn test_standardization32_transform_float() {
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let result = standardizer.transform(&Value::FloatValue(60.0)).unwrap();
        assert_eq!(result, vec![1.0]);

        let result = standardizer.transform(&Value::FloatValue(40.0)).unwrap();
        assert_eq!(result, vec![-1.0]);

        let result = standardizer.transform(&Value::FloatValue(50.0)).unwrap();
        assert_eq!(result, vec![0.0]);

        let result = standardizer.transform(&Value::IntegerValue(60)).unwrap();
        assert_eq!(result, vec![1.0]);
    }

    /// Test float array Test integer array
    #[test]
    fn test_standardization32_transform_array() {
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let float_array = hushar::hushar_proto::FloatArray {
            values: vec![40.0, 50.0, 60.0],
        };

        let result = standardizer
            .transform(&Value::FloatArray(float_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);

        let int_array = hushar::hushar_proto::IntegerArray {
            values: vec![40, 50, 60],
        };

        let result = standardizer
            .transform(&Value::IntegerArray(int_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    /// Valid Invalid: std_dev <= 0 Invalid: empty default_val
    #[test]
    fn test_standardization32_validate() {
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_ok());

        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 0.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_err());

        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![],
        };

        assert!(standardizer.validate("test_feature").is_err());
    }

    /// Test standardization Test long value
    #[test]
    fn test_standardization64_transform_double() {
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let result = standardizer.transform(&Value::DoubleValue(60.0)).unwrap();
        assert_eq!(result, vec![1.0]);

        let result = standardizer.transform(&Value::DoubleValue(40.0)).unwrap();
        assert_eq!(result, vec![-1.0]);

        let result = standardizer.transform(&Value::DoubleValue(50.0)).unwrap();
        assert_eq!(result, vec![0.0]);

        let result = standardizer.transform(&Value::LongValue(60)).unwrap();
        assert_eq!(result, vec![1.0]);
    }

    /// Test double array Test long array
    #[test]
    fn test_standardization64_transform_array() {
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let double_array = hushar::hushar_proto::DoubleArray {
            values: vec![40.0, 50.0, 60.0],
        };

        let result = standardizer
            .transform(&Value::DoubleArray(double_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);

        let long_array = hushar::hushar_proto::LongArray {
            values: vec![40, 50, 60],
        };

        let result = standardizer
            .transform(&Value::LongArray(long_array))
            .unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    /// Valid Invalid: std_dev <= 0 Invalid: empty default_val
    #[test]
    fn test_standardization64_validate() {
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_ok());

        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 0.0,
            default_val: vec![0.0],
        };

        assert!(standardizer.validate("test_feature").is_err());

        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![],
        };

        assert!(standardizer.validate("test_feature").is_err());
    }

    /// Test cases for invalid data types for each transformation Embedding with
    /// non-string types MinMaxScaling32 with string type MinMaxScaling64 with string
    /// type OneHotEncoding with numeric types Standardization32 with string type
    /// Standardization64 with string type
    #[test]
    fn test_transform_invalid_data_types() {
        let embedding = Embedding {
            embeddings: HashMap::new(),
            default_val: vec![0.0],
        };

        assert!(embedding.transform(&Value::FloatValue(1.0)).is_err());

        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(
            scaler
                .transform(&Value::StringValue("test".to_string()))
                .is_err()
        );

        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(
            scaler
                .transform(&Value::StringValue("test".to_string()))
                .is_err()
        );

        let one_hot = OneHotEncoding {
            categories: vec!["cat".to_string()],
            default_val: vec![0.0],
        };

        assert!(one_hot.transform(&Value::FloatValue(1.0)).is_err());

        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(
            standardizer
                .transform(&Value::StringValue("test".to_string()))
                .is_err()
        );

        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(
            standardizer
                .transform(&Value::StringValue("test".to_string()))
                .is_err()
        );
    }

    /// Test existing embedding Test non-existent embedding
    #[test]
    fn test_embedding_helper() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);

        let default_val = vec![0.0, 0.0, 0.0];

        let result = embedding(&"cat".to_string(), &embeddings, &default_val);
        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        let result = embedding(&"dog".to_string(), &embeddings, &default_val);
        assert_eq!(result, default_val);
    }

    /// Test within range Test at boundaries Test outside range
    #[test]
    fn test_min_max_scaling_32_helper() {
        let min = 0.0;
        let max = 10.0;

        let result = min_max_scaling_32(5.0, &min, &max);
        assert_eq!(result, 0.5);

        let result = min_max_scaling_32(0.0, &min, &max);
        assert_eq!(result, 0.0);

        let result = min_max_scaling_32(10.0, &min, &max);
        assert_eq!(result, 1.0);

        let result = min_max_scaling_32(-5.0, &min, &max);
        assert_eq!(result, -0.5);

        let result = min_max_scaling_32(15.0, &min, &max);
        assert_eq!(result, 1.5);
    }

    /// Test within range Test at boundaries Test outside range
    #[test]
    fn test_min_max_scaling_64_helper() {
        let min = 0.0;
        let max = 10.0;

        let result = min_max_scaling_64(5.0, &min, &max);
        assert_eq!(result, 0.5);

        let result = min_max_scaling_64(0.0, &min, &max);
        assert_eq!(result, 0.0);

        let result = min_max_scaling_64(10.0, &min, &max);
        assert_eq!(result, 1.0);

        let result = min_max_scaling_64(-5.0, &min, &max);
        assert_eq!(result, -0.5);

        let result = min_max_scaling_64(15.0, &min, &max);
        assert_eq!(result, 1.5);
    }

    /// Test existing categories Test non-existent category
    #[test]
    fn test_one_hot_encoding_helper() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        let result = one_hot_encoding(&"cat".to_string(), &categories, &default_val);
        assert_eq!(result, vec![1.0, 0.0, 0.0]);

        let result = one_hot_encoding(&"dog".to_string(), &categories, &default_val);
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        let result = one_hot_encoding(&"fish".to_string(), &categories, &default_val);
        assert_eq!(result, vec![0.0, 0.0, 1.0]);

        let result = one_hot_encoding(&"bird".to_string(), &categories, &default_val);
        assert_eq!(result, default_val);
    }

    /// Test standardization
    #[test]
    fn test_standardize_32_helper() {
        let mean = 50.0;
        let std_dev = 10.0;

        let result = standardize_32(40.0, &mean, &std_dev);
        assert_eq!(result, -1.0);

        let result = standardize_32(50.0, &mean, &std_dev);
        assert_eq!(result, 0.0);

        let result = standardize_32(60.0, &mean, &std_dev);
        assert_eq!(result, 1.0);

        let result = standardize_32(70.0, &mean, &std_dev);
        assert_eq!(result, 2.0);
    }

    /// Test standardization
    #[test]
    fn test_standardize_64_helper() {
        let mean = 50.0;
        let std_dev = 10.0;

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
