// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

pub(crate) mod backend;
pub(crate) mod batch;
pub(crate) mod input_builder;
pub(crate) mod neuron_backend;
pub(crate) mod onnx_backend;
pub(crate) mod proto_columns;
pub(crate) mod scoring;
pub(crate) mod session_pool;
pub(crate) mod traffic_split;
pub(crate) mod transformations;

pub(crate) use backend::InferenceBackend;
pub(crate) use session_pool::SessionPool;

/// Error type for the inference path.
///
/// `Send + Sync` is required: inference runs on a blocking thread and its result crosses
/// back to the async worker that writes the response.
pub(crate) type InferenceError = Box<dyn std::error::Error + Send + Sync>;

pub(crate) use neuron_backend::NeuronBackend;
pub(crate) use onnx_backend::OnnxRuntimeBackend;

/// Time spent in each stage of one batch, in microseconds.
///
/// # Fields
///
/// * `vec_time` - building the input batch from the request's features
/// * `tensor_time` - handing that batch to the backend
/// * `inference_time` - the engine call itself
#[derive(Debug)]
pub(crate) struct InferenceMicros {
    pub vec_time: u128,
    pub tensor_time: u128,
    pub inference_time: u128,
}
