// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Vectorization module provides the functions to process the input features into a vector input.

use hushar_proto::hushar::DataType;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use crate::config::vectorization_config::VectorizationConfig;

/// Creates a vector for the provided feature values in the order provided from the configuration.
///
/// 1. Creates a mutable vector with expected size of feat_len.
/// 2. Process the input feature data in the order provided by the configuration.
/// 3. Checks if the result size is as expected.
///
/// # Arguments
/// * `features` - The feature value to be vectorized
/// * `vec_config` - The vectorization configuration
/// * `feat_len` - The expected lenght of the feat vector == model input size.
///
/// # Returns
/// * `Result<Vec<OutputRow>, Box<dyn Error>>` - Vector of output rows or an error
pub(crate) fn vectorize_feats(
    features: &HashMap<String, DataType>,
    vec_config: &Arc<VectorizationConfig>,
    feat_len: &usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut result = Vec::with_capacity(*feat_len);

    for feature_name in &vec_config.feature_order {
        let transformation = vec_config
            .feature_transformations
            .get(feature_name)
            .ok_or_else(|| format!("No transformation found for feature: {}", feature_name))?;

        let transformed = if let Some(feat_val) = features.get(feature_name) {
            if let Some(feat_val) = &feat_val.data_type {
                transformation.transform(feat_val)?
            } else {
                transformation.get_default_val().to_vec()
            }
        } else {
            transformation.get_default_val().to_vec()
        };

        result.extend(transformed);
    }

    if result.len() != *feat_len {
        return Err(format!(
            "Expected feature vector length {} but got {}",
            feat_len,
            result.len()
        )
        .into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hushar_proto::hushar::data_type;
    use mockall::mock;
    use mockall::predicate::*;
    use std::sync::Arc;

    use crate::inference::transformations::Transformation;

    // Mock the Transformation trait for testing
    mock! {
        #[derive(Clone, Debug)]
        pub Transformation {}
        impl Transformation for Transformation {
            fn get_default_val(&self) -> &[f32];
            fn transform(&self, feat_val: &data_type::DataType) -> Result<Vec<f32>, Box<dyn Error>>;
            fn validate(&self, name: &str) -> Result<(), Box<dyn Error>>;
        }
    }

    #[test]
    fn test_vectorize_feats_success() {
        // Create a mock transformation
        let mut mock_transform1 = MockTransformation::new();
        mock_transform1
            .expect_transform()
            .returning(|_| Ok(vec![1.0, 2.0]));
        mock_transform1
            .expect_get_default_val()
            .return_const(vec![0.0, 0.0]);

        let mut mock_transform2 = MockTransformation::new();
        mock_transform2
            .expect_transform()
            .returning(|_| Ok(vec![3.0]));
        mock_transform2
            .expect_get_default_val()
            .return_const(vec![0.0]);

        // Setup test data
        let mut features = HashMap::new();
        features.insert(
            "feat1".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );
        features.insert(
            "feat2".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );

        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "feat1".to_string(),
            Box::new(mock_transform1) as Box<dyn Transformation>,
        );
        feature_transformations.insert(
            "feat2".to_string(),
            Box::new(mock_transform2) as Box<dyn Transformation>,
        );

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec!["feat1".to_string(), "feat2".to_string()],
        });

        // Expected feature vector length is 3 (2 from first transform + 1 from second)
        let feat_len = 3;

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert results
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
        assert_eq!(vec.len(), feat_len);
    }

    #[test]
    fn test_vectorize_feats_missing_feature() {
        // Create a mock transformation
        let mut mock_transform1 = MockTransformation::new();
        mock_transform1
            .expect_get_default_val()
            .return_const(vec![0.0, 0.0]);

        let mut mock_transform2 = MockTransformation::new();
        mock_transform2
            .expect_transform()
            .returning(|_| Ok(vec![3.0]));
        mock_transform2
            .expect_get_default_val()
            .return_const(vec![0.0]);

        // Setup test data - missing feat1
        let mut features = HashMap::new();
        features.insert(
            "feat2".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );

        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "feat1".to_string(),
            Box::new(mock_transform1) as Box<dyn Transformation>,
        );
        feature_transformations.insert(
            "feat2".to_string(),
            Box::new(mock_transform2) as Box<dyn Transformation>,
        );

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec!["feat1".to_string(), "feat2".to_string()],
        });

        // Expected feature vector length is 3 (2 from first transform + 1 from second)
        let feat_len = 3;

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert results
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec, vec![0.0, 0.0, 3.0]); // Default values for feat1
        assert_eq!(vec.len(), feat_len);
    }

    #[test]
    fn test_vectorize_feats_null_feature_value() {
        // Create mock transformations
        let mut mock_transform = MockTransformation::new();
        mock_transform
            .expect_get_default_val()
            .return_const(vec![0.5, 0.5]);

        // Setup test data with null value
        let mut features = HashMap::new();
        features.insert("feat1".to_string(), DataType { data_type: None });

        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "feat1".to_string(),
            Box::new(mock_transform) as Box<dyn Transformation>,
        );

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec!["feat1".to_string()],
        });

        // Expected feature vector length
        let feat_len = 2;

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert results
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec, vec![0.5, 0.5]); // Default values for null feature
        assert_eq!(vec.len(), feat_len);
    }

    #[test]
    fn test_vectorize_feats_transform_error() {
        // Create a failing transformation
        let mut mock_transform = MockTransformation::new();
        mock_transform
            .expect_transform()
            .returning(|_| Err("Transformation error".into()));

        // Setup test data
        let mut features = HashMap::new();
        features.insert(
            "feat1".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );

        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "feat1".to_string(),
            Box::new(mock_transform) as Box<dyn Transformation>,
        );

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec!["feat1".to_string()],
        });

        let feat_len = 1;

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert error
        assert!(result.is_err());
    }

    #[test]
    fn test_vectorize_feats_missing_transformation() {
        // Setup test data with a feature not having a transformation
        let mut features = HashMap::new();
        features.insert(
            "feat1".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );

        let feature_transformations = HashMap::new(); // Empty transformations

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec!["feat1".to_string()],
        });

        let feat_len = 1;

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert error
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No transformation found for feature: feat1"));
    }

    #[test]
    fn test_vectorize_feats_incorrect_length() {
        // Create a mock transformation
        let mut mock_transform = MockTransformation::new();
        mock_transform
            .expect_transform()
            .returning(|_| Ok(vec![1.0, 2.0]));

        // Setup test data
        let mut features = HashMap::new();
        features.insert(
            "feat1".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );

        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "feat1".to_string(),
            Box::new(mock_transform) as Box<dyn Transformation>,
        );

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec!["feat1".to_string()],
        });

        // Expected feature vector length is incorrect
        let feat_len = 3; // Our transform returns 2 elements

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert error
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Expected feature vector length 3 but got 2"));
    }

    #[test]
    fn test_vectorize_feats_empty_feature_order() {
        // Setup test data with empty feature order
        let features = HashMap::new();
        let feature_transformations = HashMap::new();

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec![],
        });

        let feat_len = 0;

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert results
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_vectorize_feats_multiple_features() {
        // Create mock transformations
        let mut mock_transform1 = MockTransformation::new();
        mock_transform1
            .expect_transform()
            .returning(|_| Ok(vec![1.0]));

        let mut mock_transform2 = MockTransformation::new();
        mock_transform2
            .expect_transform()
            .returning(|_| Ok(vec![2.0, 3.0]));

        let mut mock_transform3 = MockTransformation::new();
        mock_transform3
            .expect_transform()
            .returning(|_| Ok(vec![4.0, 5.0, 6.0]));

        // Setup test data
        let mut features = HashMap::new();
        features.insert(
            "feat1".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );
        features.insert(
            "feat2".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );
        features.insert(
            "feat3".to_string(),
            DataType {
                data_type: Some(data_type::DataType::IntegerValue(45)),
            },
        );

        let mut feature_transformations = HashMap::new();
        feature_transformations.insert(
            "feat1".to_string(),
            Box::new(mock_transform1) as Box<dyn Transformation>,
        );
        feature_transformations.insert(
            "feat2".to_string(),
            Box::new(mock_transform2) as Box<dyn Transformation>,
        );
        feature_transformations.insert(
            "feat3".to_string(),
            Box::new(mock_transform3) as Box<dyn Transformation>,
        );

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: vec![
                "feat1".to_string(),
                "feat2".to_string(),
                "feat3".to_string(),
            ],
        });

        // Expected feature vector length is 6 (1 + 2 + 3)
        let feat_len = 6;

        // Call the function
        let result = vectorize_feats(&features, &vec_config, &feat_len);

        // Assert results
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(vec.len(), feat_len);
    }
}
