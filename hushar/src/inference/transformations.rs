// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Feature transformations that convert request data types into ML model inputs.
//!
//! - Embedding: maps string values to pre-trained vector embeddings
//! - Identity: passes through numerical values with minimal transformation
//! - MinMaxScaling: scales numerical values to a specified range
//! - OneHotEncoding: converts categorical values to one-hot encoded vectors
//! - Standardization: standardizes numerical values using mean and standard deviation
//!
//! Each transformation reads one feature's values, borrowed from the column that holds
//! them, and appends what they contribute to the caller's buffer. A scalar feature is a
//! slice of one.

use crate::inference::batch::FeatureSlice;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Turns one row of a feature into the numbers a model input takes.
pub trait Transformation: std::fmt::Debug + Send + Sync {
    /// Returns the default value to use when transformation is not possible.
    fn get_default_val(&self) -> &[f32];

    /// Appends what this row contributes to the row being built.
    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError>;

    /// The same transformation, in double precision.
    ///
    /// The default widens the `f32` result, so it is no more precise than that path. The
    /// 64-bit scalers and [`Identity`] override it to compute in `f64` throughout.
    fn transform_f64(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f64>,
    ) -> Result<(), crate::inference::InferenceError> {
        let mut narrow = Vec::new();
        self.transform(values, &mut narrow)?;
        out.extend(narrow.into_iter().map(f64::from));
        Ok(())
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

/// Says what arrived when a transformation cannot use it.
fn wrong_type(what: &str, got: FeatureSlice<'_>) -> crate::inference::InferenceError {
    format!("Invalid data type for {what}: got {}", got.describe()).into()
}

/// Maps string values to vector embeddings using a pre-defined embedding table.
///
/// A string that is not in the table contributes `default_val`.
///
/// # Fields
///
/// * `default_val` - used when a string is not in the table
/// * `embeddings` - each string value to its vector
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

    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::Str(v) => {
                for s in v {
                    match self.embeddings.get(s) {
                        Some(e) => out.extend_from_slice(e),
                        None => out.extend_from_slice(&self.default_val),
                    }
                }
                Ok(())
            }
            other => Err(wrong_type("Embedding", other)),
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

/// Passes through numerical values, cast to the model input's type.
///
/// # Fields
///
/// * `default_val` - used when the feature value is absent
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Identity {
    pub default_val: Vec<f32>,
}

impl Transformation for Identity {
    fn get_default_val(&self) -> &[f32] {
        &self.default_val
    }

    /// Parses text as a number, so a caller sending a number as a string still reaches
    /// a numeric input.
    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F32(v) => out.extend_from_slice(v),
            FeatureSlice::F64(v) => out.extend(v.iter().map(|x| *x as f32)),
            FeatureSlice::I32(v) => out.extend(v.iter().map(|x| *x as f32)),
            FeatureSlice::I64(v) => out.extend(v.iter().map(|x| *x as f32)),
            FeatureSlice::Str(v) => {
                let mut errors = Vec::new();
                for s in v {
                    match s.parse::<f32>() {
                        Ok(f) => out.push(f),
                        Err(_) => errors.push(s.clone()),
                    }
                }
                if !errors.is_empty() {
                    return Err(format!(
                        "Error in parsing text as a number.\n {}",
                        errors.join(", ")
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    fn validate(&self, name: &str) -> Result<(), crate::inference::InferenceError> {
        if self.default_val.is_empty() {
            return Err(format!("Identity for {}: default_val must not be empty", name).into());
        }
        Ok(())
    }

    /// Passes the value through without ever narrowing it.
    ///
    /// A `double` feature reaching a `double` input keeps its low bits.
    fn transform_f64(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f64>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F32(v) => out.extend(v.iter().copied().map(f64::from)),
            FeatureSlice::F64(v) => out.extend_from_slice(v),
            FeatureSlice::I32(v) => out.extend(v.iter().copied().map(f64::from)),
            FeatureSlice::I64(v) => out.extend(v.iter().map(|x| *x as f64)),
            FeatureSlice::Str(v) => {
                let mut errors = Vec::new();
                for s in v {
                    match s.parse::<f64>() {
                        Ok(f) => out.push(f),
                        Err(_) => errors.push(s.clone()),
                    }
                }
                if !errors.is_empty() {
                    return Err(format!(
                        "Error in parsing text as a number.\n {}",
                        errors.join(", ")
                    )
                    .into());
                }
            }
        }
        Ok(())
    }
}

/// Scales a 32-bit value into `[0, 1]` as `(value - min) / (max - min)`.
///
/// # Fields
///
/// * `default_val` - used when the value cannot be scaled
/// * `min`, `max` - the original scale
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

    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F32(v) => out.extend(
                v.iter()
                    .map(|x| min_max_scaling_32(*x, &self.min, &self.max)),
            ),
            FeatureSlice::I32(v) => out.extend(
                v.iter()
                    .map(|x| min_max_scaling_32(*x as f32, &self.min, &self.max)),
            ),
            other => return Err(wrong_type("MinMaxScaling32", other)),
        }
        Ok(())
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
/// # Fields
///
/// * `default_val` - used when the value cannot be scaled
/// * `min`, `max` - the original scale
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

    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F64(v) => out.extend(
                v.iter()
                    .map(|x| min_max_scaling_64(*x, &self.min, &self.max) as f32),
            ),
            FeatureSlice::I64(v) => out.extend(
                v.iter()
                    .map(|x| min_max_scaling_64(*x as f64, &self.min, &self.max) as f32),
            ),
            other => return Err(wrong_type("MinMaxScaling64", other)),
        }
        Ok(())
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
    /// [`MinMaxScaling64::transform`] computes in `f64` and then narrows; this omits the
    /// narrowing.
    fn transform_f64(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f64>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F64(v) => out.extend(
                v.iter()
                    .map(|x| min_max_scaling_64(*x, &self.min, &self.max)),
            ),
            FeatureSlice::I64(v) => out.extend(
                v.iter()
                    .map(|x| min_max_scaling_64(*x as f64, &self.min, &self.max)),
            ),
            other => return Err(wrong_type("MinMaxScaling64", other)),
        }
        Ok(())
    }
}

/// Converts categorical string values into one-hot encoded vectors.
///
/// Each string contributes `categories.len()` values, all 0 except a 1 at the position of
/// its category. `categories` must be sorted, since the lookup is a binary search. An
/// unknown category contributes `default_val`.
///
/// # Fields
///
/// * `default_val` - used for an unknown category
/// * `categories` - the categories, which must be sorted
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

    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::Str(v) => {
                for s in v {
                    match self.categories.binary_search(s) {
                        Ok(index) => {
                            let at = out.len();
                            out.resize(at + self.categories.len(), 0.0);
                            out[at + index] = 1.0;
                        }
                        Err(_) => out.extend_from_slice(&self.default_val),
                    }
                }
                Ok(())
            }
            other => Err(wrong_type("OneHotEncoding", other)),
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

/// Standardizes 32-bit values as `(value - mean) / std_dev`.
///
/// # Fields
///
/// * `default_val` - used when the value cannot be standardized
/// * `mean`, `std_dev` - the distribution
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

    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F32(v) => out.extend(
                v.iter()
                    .map(|x| standardize_32(*x, &self.mean, &self.std_dev)),
            ),
            FeatureSlice::I32(v) => out.extend(
                v.iter()
                    .map(|x| standardize_32(*x as f32, &self.mean, &self.std_dev)),
            ),
            other => return Err(wrong_type("Standardization32", other)),
        }
        Ok(())
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

/// Standardizes 64-bit values as `(value - mean) / std_dev`.
///
/// # Fields
///
/// * `default_val` - used when the value cannot be standardized
/// * `mean`, `std_dev` - the distribution
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

    fn transform(
        &self,
        values: FeatureSlice<'_>,
        out: &mut Vec<f32>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F64(v) => out.extend(
                v.iter()
                    .map(|x| standardize_64(*x, &self.mean, &self.std_dev) as f32),
            ),
            FeatureSlice::I64(v) => out.extend(
                v.iter()
                    .map(|x| standardize_64(*x as f64, &self.mean, &self.std_dev) as f32),
            ),
            other => return Err(wrong_type("Standardization64", other)),
        }
        Ok(())
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
        values: FeatureSlice<'_>,
        out: &mut Vec<f64>,
    ) -> Result<(), crate::inference::InferenceError> {
        match values {
            FeatureSlice::F64(v) => out.extend(
                v.iter()
                    .map(|x| standardize_64(*x, &self.mean, &self.std_dev)),
            ),
            FeatureSlice::I64(v) => out.extend(
                v.iter()
                    .map(|x| standardize_64(*x as f64, &self.mean, &self.std_dev)),
            ),
            other => return Err(wrong_type("Standardization64", other)),
        }
        Ok(())
    }
}

fn min_max_scaling_32(val: f32, min: &f32, max: &f32) -> f32 {
    (val - min) / (max - min)
}

fn min_max_scaling_64(val: f64, min: &f64, max: &f64) -> f64 {
    (val - min) / (max - min)
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
    use crate::inference::InferenceError;
    use std::collections::HashMap;

    /// Owns one feature's values so a test can borrow them as a `FeatureSlice`.
    enum V {
        F32(Vec<f32>),
        F64(Vec<f64>),
        I32(Vec<i32>),
        I64(Vec<i64>),
        Str(Vec<String>),
    }

    impl V {
        fn slice(&self) -> FeatureSlice<'_> {
            match self {
                V::F32(v) => FeatureSlice::F32(v),
                V::F64(v) => FeatureSlice::F64(v),
                V::I32(v) => FeatureSlice::I32(v),
                V::I64(v) => FeatureSlice::I64(v),
                V::Str(v) => FeatureSlice::Str(v),
            }
        }
    }

    /// Calls a transformation and collects what it appended.
    trait TestTransform {
        fn apply(&self, v: &V) -> Result<Vec<f32>, InferenceError>;
        fn apply_f64(&self, v: &V) -> Result<Vec<f64>, InferenceError>;
    }

    impl<T: Transformation + ?Sized> TestTransform for T {
        fn apply(&self, v: &V) -> Result<Vec<f32>, InferenceError> {
            let mut out = Vec::new();
            self.transform(v.slice(), &mut out)?;
            Ok(out)
        }
        fn apply_f64(&self, v: &V) -> Result<Vec<f64>, InferenceError> {
            let mut out = Vec::new();
            self.transform_f64(v.slice(), &mut out)?;
            Ok(out)
        }
    }

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

    /// The `f64` path keeps a double's low bits, where the `f32` path cannot.
    #[test]
    fn the_double_path_does_not_narrow_through_a_float() {
        let identity = Identity {
            default_val: vec![0.0],
        };
        let sent = 0.1_f64;

        assert_eq!(
            identity.apply_f64(&V::F64(vec![sent])).expect("a double"),
            vec![sent],
            "Identity overrides transform_f64, so the low bits survive"
        );
        assert_ne!(
            identity.apply(&V::F64(vec![sent])).expect("a double")[0] as f64,
            sent,
            "the f32 path cannot hold 0.1, which is the reason the override exists"
        );

        let scaler = MinMaxScaling64 {
            default_val: vec![0.0],
            min: 0.0,
            max: 3.0,
        };
        let scaled = scaler.apply_f64(&V::F64(vec![1.0])).expect("in range");
        assert_eq!(scaled, vec![1.0 / 3.0], "divided in f64, not in f32");
        assert_ne!(
            scaled[0],
            f64::from(1.0_f32 / 3.0_f32),
            "and not the same number the f32 path would give"
        );
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

        let result = embedding.apply(&V::Str(vec!["cat".to_string()])).unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        let result = embedding.apply(&V::Str(vec!["bird".to_string()])).unwrap();
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

        let string_values = vec!["cat".to_string(), "bird".to_string(), "dog".to_string()];

        let result = embedding.apply(&V::Str(string_values)).unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.0, 0.0, 0.3, 0.4]);
    }

    /// Validation rejects an empty embedding table and an empty default value.
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

    /// Validation rejects an empty default value.
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
        let input = V::F64(vec![42.5]);
        let result = identity.apply(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_float_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = V::F32(vec![42.5]);
        let result = identity.apply(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_integer_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = V::I32(vec![42]);
        let result = identity.apply(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.0_f32);
    }

    #[test]
    fn test_transform_long_value() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = V::I64(vec![42_i64]);
        let result = identity.apply(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.0_f32);
    }

    #[test]
    fn test_transform_string_value_valid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = V::Str(vec!["42.5".to_string()]);
        let result = identity.apply(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_float_eq(result[0], 42.5_f32);
    }

    #[test]
    fn test_transform_string_value_invalid() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = V::Str(vec!["not_a_number".to_string()]);
        let result = identity.apply(&input);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Error in parsing text as a number")
        );
    }

    #[test]
    fn test_transform_double_array() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = V::F64(vec![1.1, 2.2, 3.3]);
        let result = identity.apply(&input).unwrap();
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
        let input = V::F32(vec![1.1, 2.2, 3.3]);
        let result = identity.apply(&input).unwrap();
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
        let input = V::I32(vec![1, 2, 3]);
        let result = identity.apply(&input).unwrap();
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
        let input = V::I64(vec![1, 2, 3]);
        let result = identity.apply(&input).unwrap();
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
        let input = V::Str(vec![
            "1.1".to_string(),
            "2.2".to_string(),
            "3.3".to_string(),
        ]);
        let result = identity.apply(&input).unwrap();
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
        let input = V::Str(vec![
            "1.1".to_string(),
            "not_a_number".to_string(),
            "3.3".to_string(),
        ]);
        let result = identity.apply(&input);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Error in parsing text as a number")
        );
    }

    #[test]
    fn test_transform_string_array_empty() {
        let identity = Identity {
            default_val: vec![1.0, 2.0, 3.0],
        };
        let input = V::Str(vec![]);
        let result = identity.apply(&input).unwrap();
        assert_eq!(result.len(), 0);
    }

    /// Scales floats and integers, inside the range and outside it.
    #[test]
    fn test_min_max_scaling32_transform_float() {
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let result = scaler.apply(&V::F32(vec![5.0])).unwrap();
        assert_eq!(result, vec![0.5]);

        let result = scaler.apply(&V::F32(vec![-5.0])).unwrap();
        assert_eq!(result, vec![-0.5]);

        let result = scaler.apply(&V::F32(vec![15.0])).unwrap();
        assert_eq!(result, vec![1.5]);

        let result = scaler.apply(&V::I32(vec![5])).unwrap();
        assert_eq!(result, vec![0.5]);
    }

    /// Scales every value of a float or integer column.
    #[test]
    fn test_min_max_scaling32_transform_array() {
        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let float_values = vec![0.0, 5.0, 10.0];

        let result = scaler.apply(&V::F32(float_values)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);

        let int_values = vec![0, 5, 10];

        let result = scaler.apply(&V::I32(int_values)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);
    }

    /// Validation rejects `min >= max` and an empty default value.
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

    /// Scales doubles and longs, inside the range and outside it.
    #[test]
    fn test_min_max_scaling64_transform_double() {
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let result = scaler.apply(&V::F64(vec![5.0])).unwrap();
        assert_eq!(result, vec![0.5]);

        let result = scaler.apply(&V::F64(vec![-5.0])).unwrap();
        assert_eq!(result, vec![-0.5]);

        let result = scaler.apply(&V::F64(vec![15.0])).unwrap();
        assert_eq!(result, vec![1.5]);

        let result = scaler.apply(&V::I64(vec![5])).unwrap();
        assert_eq!(result, vec![0.5]);
    }

    /// Scales every value of a double or long column.
    #[test]
    fn test_min_max_scaling64_transform_array() {
        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        let double_values = vec![0.0, 5.0, 10.0];

        let result = scaler.apply(&V::F64(double_values)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);

        let long_values = vec![0, 5, 10];

        let result = scaler.apply(&V::I64(long_values)).unwrap();
        assert_eq!(result, vec![0.0, 0.5, 1.0]);
    }

    /// Validation rejects `min >= max` and an empty default value.
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

    /// A known category encodes to its position; an unknown one takes the default.
    #[test]
    fn test_one_hot_encoding_transform_string_value() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        let one_hot = OneHotEncoding {
            categories,
            default_val: default_val.clone(),
        };

        let result = one_hot.apply(&V::Str(vec!["cat".to_string()])).unwrap();
        assert_eq!(result, vec![1.0, 0.0, 0.0]);

        let result = one_hot.apply(&V::Str(vec!["dog".to_string()])).unwrap();
        assert_eq!(result, vec![0.0, 1.0, 0.0]);

        let result = one_hot.apply(&V::Str(vec!["bird".to_string()])).unwrap();
        assert_eq!(result, default_val);
    }

    /// A string column encodes row by row, an unknown category taking the default.
    #[test]
    fn test_one_hot_encoding_transform_string_array() {
        let categories = vec!["cat".to_string(), "dog".to_string(), "fish".to_string()];
        let default_val = vec![0.0, 0.0, 0.0];

        let one_hot = OneHotEncoding {
            categories,
            default_val: default_val.clone(),
        };

        let string_values = vec!["cat".to_string(), "bird".to_string(), "fish".to_string()];

        let result = one_hot.apply(&V::Str(string_values)).unwrap();
        assert_eq!(result, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// Validation rejects empty categories and an empty default value.
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

    /// Standardizes floats and integers against the configured distribution.
    #[test]
    fn test_standardization32_transform_float() {
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let result = standardizer.apply(&V::F32(vec![60.0])).unwrap();
        assert_eq!(result, vec![1.0]);

        let result = standardizer.apply(&V::F32(vec![40.0])).unwrap();
        assert_eq!(result, vec![-1.0]);

        let result = standardizer.apply(&V::F32(vec![50.0])).unwrap();
        assert_eq!(result, vec![0.0]);

        let result = standardizer.apply(&V::I32(vec![60])).unwrap();
        assert_eq!(result, vec![1.0]);
    }

    /// Standardizes every value of a float or integer column.
    #[test]
    fn test_standardization32_transform_array() {
        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let float_values = vec![40.0, 50.0, 60.0];

        let result = standardizer.apply(&V::F32(float_values)).unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);

        let int_values = vec![40, 50, 60];

        let result = standardizer.apply(&V::I32(int_values)).unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    /// Validation rejects a non-positive `std_dev` and an empty default value.
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

    /// Standardizes doubles and longs against the configured distribution.
    #[test]
    fn test_standardization64_transform_double() {
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let result = standardizer.apply(&V::F64(vec![60.0])).unwrap();
        assert_eq!(result, vec![1.0]);

        let result = standardizer.apply(&V::F64(vec![40.0])).unwrap();
        assert_eq!(result, vec![-1.0]);

        let result = standardizer.apply(&V::F64(vec![50.0])).unwrap();
        assert_eq!(result, vec![0.0]);

        let result = standardizer.apply(&V::I64(vec![60])).unwrap();
        assert_eq!(result, vec![1.0]);
    }

    /// Standardizes every value of a double or long column.
    #[test]
    fn test_standardization64_transform_array() {
        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        let double_values = vec![40.0, 50.0, 60.0];

        let result = standardizer.apply(&V::F64(double_values)).unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);

        let long_values = vec![40, 50, 60];

        let result = standardizer.apply(&V::I64(long_values)).unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    /// Validation rejects a non-positive `std_dev` and an empty default value.
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

    /// Every transformation errors on a feature type it cannot read.
    #[test]
    fn test_transform_invalid_data_types() {
        let embedding = Embedding {
            embeddings: HashMap::new(),
            default_val: vec![0.0],
        };

        assert!(embedding.apply(&V::F32(vec![1.0])).is_err());

        let scaler = MinMaxScaling32 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.apply(&V::Str(vec!["test".to_string()])).is_err());

        let scaler = MinMaxScaling64 {
            min: 0.0,
            max: 10.0,
            default_val: vec![0.5],
        };

        assert!(scaler.apply(&V::Str(vec!["test".to_string()])).is_err());

        let one_hot = OneHotEncoding {
            categories: vec!["cat".to_string()],
            default_val: vec![0.0],
        };

        assert!(one_hot.apply(&V::F32(vec![1.0])).is_err());

        let standardizer = Standardization32 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(
            standardizer
                .apply(&V::Str(vec!["test".to_string()]))
                .is_err()
        );

        let standardizer = Standardization64 {
            mean: 50.0,
            std_dev: 10.0,
            default_val: vec![0.0],
        };

        assert!(
            standardizer
                .apply(&V::Str(vec!["test".to_string()]))
                .is_err()
        );
    }

    /// A known string gets its vector; an unknown one gets the default.
    #[test]
    fn an_unknown_string_embeds_as_the_default() {
        let mut embeddings = HashMap::new();
        embeddings.insert("cat".to_string(), vec![0.1, 0.2, 0.3]);
        let default_val = vec![0.0, 0.0, 0.0];
        let t = Embedding {
            default_val: default_val.clone(),
            embeddings,
        };

        assert_eq!(
            t.apply(&V::Str(vec!["cat".to_string()])).expect("known"),
            vec![0.1, 0.2, 0.3]
        );
        assert_eq!(
            t.apply(&V::Str(vec!["dog".to_string()])).expect("unknown"),
            default_val
        );
    }

    /// The helper scales values inside the range, at its bounds, and outside it.
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

    /// The helper scales values inside the range, at its bounds, and outside it.
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

    /// Each category sets its own position and nothing else; an unknown one falls back.
    #[test]
    fn one_hot_sets_exactly_its_own_position() {
        let default_val = vec![0.0, 0.0, 0.0];
        let t = OneHotEncoding {
            default_val: default_val.clone(),
            categories: vec!["cat".to_string(), "dog".to_string(), "fish".to_string()],
        };

        for (word, expected) in [
            ("cat", vec![1.0, 0.0, 0.0]),
            ("dog", vec![0.0, 1.0, 0.0]),
            ("fish", vec![0.0, 0.0, 1.0]),
        ] {
            assert_eq!(
                t.apply(&V::Str(vec![word.to_string()])).expect(word),
                expected,
                "{word}"
            );
        }
        assert_eq!(
            t.apply(&V::Str(vec!["bird".to_string()])).expect("unknown"),
            default_val
        );
    }

    /// The helper returns the value's distance from the mean in standard deviations.
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

    /// The helper returns the value's distance from the mean in standard deviations.
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
