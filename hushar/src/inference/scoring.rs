// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Scoring module provides the functions to process the input data into model predictions.
//! Provides functions to create model input vectors from the provided data,
//! and run inference on the vectorized batch of tensors.

use crate::inference::{InferenceMicros, TractRunnableModel};
use hushar::hushar_proto::{InferenceLogRow, InputRow, OutputRow};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;
use tract_onnx::prelude::*;

use crate::config::vectorization_config::VectorizationConfig;

/// Runs inference on the provided tract model graph and the provided input rows.
///
/// 1. Vectorize the input features into the order specified by the configuration.
/// 2. Create a batch of tensors for inference.
/// 3. Run the model graph for inference on the batch of input tensor.
/// 4. Process the output tensor and return the result along with time taken to process each sub-step.
///
/// # Arguments
/// * `feat_len` - The expected lenght of the feat vector == model input size.
/// * `model` - The model to run inference.
/// * `inputs` - The input batch of features to be used for inference.
/// * `vec_config` - The vectorization configuration
///
/// # Returns
/// * `Result<(Vec<OutputRow>, InferenceMicroSeconds), Box<dyn Error>>`
pub fn batch_inference(
    feat_len: &usize,
    model: &Arc<TractRunnableModel>,
    inputs: Vec<InputRow>,
    vec_config: &Arc<VectorizationConfig>,
) -> Result<(Vec<OutputRow>, Vec<InferenceLogRow>, InferenceMicros), Box<dyn Error>> {
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            Vec::new(),
            InferenceMicros {
                vec_time: 0,
                tensor_time: 0,
                inference_time: 0,
            },
        ));
    }
    let vec_start_time = Instant::now();
    let mut batch_features = Vec::with_capacity(inputs.len() * (*feat_len));
    let mut row_ids = Vec::with_capacity(inputs.len());
    let mut results = Vec::with_capacity(inputs.len());
    let mut inference_logs = Vec::with_capacity(inputs.len());

    for input in &inputs {
        let feature_vector = crate::inference::vectorization::vectorize_feats(
            &input.features,
            vec_config,
            feat_len,
        )?;
        batch_features.extend(feature_vector);
        row_ids.push(input.row_id.clone());
    }

    let tensor_start_time = Instant::now();
    let batch_tensor =
        tract_ndarray::Array2::from_shape_vec((inputs.len(), *feat_len), batch_features)
            .map_err(|e| format!("Failed to reshape batch tensor: {}", e))?
            .into_arc_tensor();
    let batch_input = TValue::Const(batch_tensor);

    let inference_start_time = Instant::now();
    let outputs = model.run(tvec!(batch_input))?;
    let inference_end_time = Instant::now();

    let output_tensor = outputs[0]
        .to_array_view::<f32>()
        .map_err(|e| format!("Failed to interpret model output: {}", e))?;

    let pred_len = output_tensor.shape()[1];
    for (i, row_id) in row_ids.iter().enumerate() {
        let mut predictions = Vec::with_capacity(pred_len);
        for j in 0..pred_len {
            predictions.push(output_tensor[[i, j]]);
        }
        results.push(OutputRow {
            row_id: row_id.clone(),
            scores: predictions.clone(),
        });
        inference_logs.push(InferenceLogRow {
            row_id: row_id.clone(),
            features: inputs.get(i).unwrap().features.clone(),
            scores: predictions,
        });
    }

    let vec_time = (tensor_start_time - vec_start_time).as_micros();
    let tensor_time = (inference_start_time - tensor_start_time).as_micros();
    let inference_time = (inference_end_time - inference_start_time).as_micros();

    Ok((
        results,
        inference_logs,
        InferenceMicros {
            vec_time,
            tensor_time,
            inference_time,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::transformations::Transformation;
    use hushar::hushar_proto::{data_type::DataType, InputRow};
    use mockall::mock;
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;
    use std::sync::Arc;

    use crate::config::vectorization_config::VectorizationConfig;

    // Mock the Transformation trait for testing
    mock! {
        #[derive(Clone, Debug)]
        pub Transformation {}
        impl Transformation for Transformation {
            fn get_default_val(&self) -> &[f32];
            fn transform(&self, feat_val: &DataType) -> Result<Vec<f32>, Box<dyn Error>>;
            fn validate(&self, name: &str) -> Result<(), Box<dyn Error>>;
        }
    }

    #[test]
    fn test_batch_inference() -> Result<(), Box<dyn Error>> {
        // Set up the test model path
        let model_path = Path::new("test-data/sigmoid_model_12.onnx");
        if !model_path.exists() {
            panic!("Test model not found at {:?}. Make sure to run the generate_test_model.py script with feature_len=12 first.", model_path);
        }

        let mut file = File::open(model_path)?;
        let mut model_bytes = Vec::new();
        file.read_to_end(&mut model_bytes)?;

        let (model, feat_len) = crate::io::model_loader::load_onnx_model(&model_bytes)?;
        let model = Arc::new(model);

        // Set up VectorizationConfig
        let mut feature_transformations = HashMap::new();
        let feature_list = vec![
            "feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "feat7", "feat8", "feat9",
            "feat10", "feat11", "feat12",
        ];

        for feat_name in &feature_list {
            let mut mock_transform = MockTransformation::new();
            mock_transform
                .expect_transform()
                .returning(|_| Ok(vec![0.56]));
            mock_transform
                .expect_get_default_val()
                .return_const(vec![0.77]);

            feature_transformations.insert(
                feat_name.to_string(),
                Box::new(mock_transform) as Box<dyn Transformation>,
            );
        }

        let vec_config = Arc::new(VectorizationConfig {
            feature_transformations,
            feature_order: feature_list.iter().map(|s| s.to_string()).collect(),
        });

        // Create test input data
        let inputs = create_test_inputs(5); // Create 5 test input rows

        // Run batch inference
        let (results, inference_logs, inference_micros) =
            batch_inference(&feat_len, &model, inputs.clone(), &vec_config)?;

        // Print time taken for each sub-step
        println!("Vectorization time: {} µs", inference_micros.vec_time);
        println!("Tensor creation time: {} µs", inference_micros.tensor_time);
        println!("Inference time: {} µs", inference_micros.inference_time);
        println!(
            "Total time: {} µs",
            inference_micros.vec_time
                + inference_micros.tensor_time
                + inference_micros.inference_time
        );

        // Verify results
        assert_eq!(
            results.len(),
            inputs.len(),
            "Number of output rows should match number of input rows"
        );

        // Verify results
        assert_eq!(
            inference_logs.len(),
            inputs.len(),
            "Number of output rows should match number of input rows"
        );

        // Each output should have 2 scores (based on the model definition in generate_test_model.py)
        for result in &results {
            assert_eq!(result.scores.len(), 2, "Each output should have 2 scores");

            // Scores should be between 0 and 1 (sigmoid output)
            for score in &result.scores {
                assert!(
                    *score >= 0.0 && *score <= 1.0,
                    "Scores should be between 0 and 1"
                );
            }
        }

        Ok(())
    }

    // Helper function to create test inputs
    fn create_test_inputs(count: usize) -> Vec<InputRow> {
        let mut inputs = Vec::with_capacity(count);

        for i in 0..count {
            let mut features = HashMap::new();

            // Add 12 features to match our model input size
            for j in 1..=12 {
                let feat_name = format!("feat{}", j);
                let value = (i as f32 * 0.1) + (j as f32 * 0.05); // Some deterministic value

                let data_type = hushar::hushar_proto::DataType {
                    data_type: Some(DataType::FloatValue(value)),
                };

                features.insert(feat_name, data_type);
            }

            inputs.push(InputRow {
                row_id: format!("row_{}", i),
                features,
            });
        }

        inputs
    }
}
