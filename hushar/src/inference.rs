// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

pub(crate) mod backend;
pub(crate) mod batch;
pub(crate) mod input_builder;
pub(crate) mod onnx_backend;
pub(crate) mod scoring;
pub(crate) mod transformations;

pub(crate) use backend::InferenceBackend;

/// Error type for the inference path.
///
/// `Send + Sync` is load-bearing, not decoration. Inference is CPU-bound and runs
/// on a blocking thread, so its result has to cross back to the async worker that
/// will write the response; a plain `Box<dyn Error>` cannot make that trip.
pub(crate) type InferenceError = Box<dyn std::error::Error + Send + Sync>;

pub(crate) use onnx_backend::OnnxRuntimeBackend;

/// Time spent in each stage of one batch, in microseconds.
///
/// Fields:
/// - `vec_time` — building the input batch from the request's features.
/// - `tensor_time` — handing that batch to the backend.
/// - `inference_time` — the engine call itself.
#[derive(Debug)]
pub(crate) struct InferenceMicros {
    pub vec_time: u128,
    pub tensor_time: u128,
    pub inference_time: u128,
}
