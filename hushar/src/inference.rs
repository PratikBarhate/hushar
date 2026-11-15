// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

pub(crate) mod backend;
pub(crate) mod batch;
pub(crate) mod execution_plan;
pub(crate) mod input_builder;
pub(crate) mod merger;
pub(crate) mod neuron_backend;
pub(crate) mod onnx_backend;
pub(crate) mod proto_columns;
pub(crate) mod scoring;
pub(crate) mod session_pool;
pub(crate) mod traffic_split;
pub(crate) mod transformations;

pub(crate) use backend::InferenceBackend;
pub(crate) use execution_plan::ExecutionPlan;
pub(crate) use session_pool::SessionPool;

/// Error type for the inference path.
///
/// `Send + Sync` is required: inference runs on a blocking thread and its result crosses
/// back to the async worker that writes the response.
pub(crate) type InferenceError = Box<dyn std::error::Error + Send + Sync>;

pub(crate) use neuron_backend::NeuronBackend;
pub(crate) use onnx_backend::OnnxRuntimeBackend;

/// What one scored request reports.
///
/// The first three are durations in microseconds and have always been published. The rest
/// are detailed figures, published only when `metrics.detailed` is set, because they
/// triple the data points a request emits and the metric path has been seen to drop
/// batches at the smaller volume.
///
/// Field order is the order [`crate::io::metrics`] publishes them in, so the two must
/// change together.
///
/// # Fields
///
/// * `vec_time` - building the input batch from the request's features
/// * `tensor_time` - handing that batch to the backend
/// * `inference_time` - the engine call itself. Across mini batches this is the longest
///   single call, not their sum, so it is a critical path rather than device work
/// * `request_time` - the request's own span, from decoded features to scores. The figure
///   `inference_time` cannot give once calls overlap or are shared between requests
/// * `lease_wait` - waiting for a session. Zero on a merged request, which holds no
///   session of its own
/// * `engine_calls` - engine calls this request became
/// * `ballast_rows` - rows computed and thrown away to fill a fixed call shape
/// * `fill_percent` - real rows as a percentage of the rows sent to the engine
/// * `queue_depth` - requests waiting for a merged call when this one joined the queue.
///   Zero on an unmerged request
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RequestReport {
    pub vec_time: u128,
    pub tensor_time: u128,
    pub inference_time: u128,
    pub request_time: u128,
    pub lease_wait: u128,
    pub engine_calls: u128,
    pub ballast_rows: u128,
    pub fill_percent: u128,
    pub queue_depth: u128,
}

impl RequestReport {
    /// Every value, in the order [`crate::io::metrics`] publishes them.
    pub(crate) fn values(&self) -> [u128; 9] {
        [
            self.vec_time,
            self.tensor_time,
            self.inference_time,
            self.request_time,
            self.lease_wait,
            self.engine_calls,
            self.ballast_rows,
            self.fill_percent,
            self.queue_depth,
        ]
    }

    /// Sets `fill_percent` from the rows kept and the rows sent.
    ///
    /// A call sent no rows has nothing to be full of, and is reported as full rather than
    /// as empty, so an idle model does not look like a wasteful one.
    pub(crate) fn set_fill(&mut self, kept: usize, sent: usize) {
        self.fill_percent = if sent == 0 {
            100
        } else {
            (kept as u128 * 100) / sent as u128
        };
    }
}
