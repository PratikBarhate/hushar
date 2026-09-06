// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Everything about one model: where it is, how it runs, and how features reach it.
//!
//! Split from the service configuration along one line: this file describes the model
//! and the hardware it runs on, and the service configuration describes the process.
//! So `execution_provider` lives here, beside the model it applies to, while the
//! service configuration keeps only the port and the per-connection limit.
//!
//! ```json
//! {
//!   "model_id": "fraud-v3",
//!   "model_path": "s3://models/fraud/v3/model.onnx",
//!   "execution_provider": "cpu",
//!   "intra_op_threads": 1,
//!   "vectorization_config": {
//!     "data_type": "float",
//!     "feature_transformations": { "age": { "type": "identity", "default_val": [0.0] } },
//!     "feature_order": ["age", "city"]
//!   }
//! }
//! ```
//!
//! `vectorization_config` is optional. Left out, every model input is fed by the
//! feature of the same name, passed through with the model's own element type --
//! which is the useful default for a model whose inputs are already named after
//! features, and needs no configuration at all.

use crate::config::vectorization_config::VectorizationConfig;
use serde::Deserialize;

/// CPU unless configured otherwise, which is the only provider guaranteed to be
/// present in every ONNX Runtime build.
fn default_execution_provider() -> String {
    "cpu".to_owned()
}

/// One thread inside each operator.
///
/// The service already saturates the machine with concurrent requests, so letting
/// ONNX Runtime fan a single operator across cores adds contention rather than
/// throughput. Override for large models served at low concurrency.
fn default_intra_op_threads() -> i32 {
    1
}

/// One model, and how to serve it.
///
/// Fields:
/// - `model_id` — recorded on every inference log row, so logs join to a model version.
/// - `model_path` — where the model file is, carrying its own URI scheme.
/// - `execution_provider` — the hardware backend. See below.
/// - `intra_op_threads` — threads ONNX Runtime may use within a single operator.
/// - `fixed_batch_size` — serve only batches of exactly this many rows. See below.
/// - `vectorization_config` — how features become model inputs, or `None` to pass each
///   input's like-named feature straight through.
///
/// # Execution providers
///
/// Accepts `cpu`, `coreml`, `coreml:cpu_and_neural_engine`, `xnnpack`,
/// `xnnpack:<threads>`, `tensorrt[:device]` for NVIDIA and `migraphx[:device]` for AMD.
/// Any other value is passed to ONNX Runtime as a provider name, so providers this
/// service does not model can still be used.
///
/// `cuda` and `rocm` are deliberately rejected: TensorRT and MIGraphX are the vendors'
/// strategic inference stacks, so a config naming the lower-level provider gets an error
/// pointing at the right one instead of quietly behaving differently.
///
/// The provider must be compiled into the `libonnxruntime` on the host; startup fails
/// with the list of available providers if it is not.
///
/// # Fixed batch size
///
/// Left out, the model's row axis must be dynamic and a request may carry any number of
/// rows. That is the default because it is what a serving path wants: the service batches
/// however many rows arrive.
///
/// Set, it declares the batch size the model's graph is built for, and a model may then
/// pin its leading dimension to the same number. One check follows at
/// load: a pinned axis disagreeing with this number is refused when the model loads,
/// because every request would then fail inside the engine on a shape mismatch.
///
/// It exists so a model that was exported with a pinned leading dimension can be served
/// at all: without this field such a model is refused when it loads, because it could
/// only ever be given batches of that one size.
///
/// **It does not constrain callers.** A request of any size is served: rows are cut into
/// batches of this many and the final short batch is padded up to it, with the padded
/// rows scored and then discarded. So the pinned shape is satisfied without a client
/// having to know the number. What it costs is work -- one row against a size of 32 runs
/// a batch of 32 -- so pin to a size near the traffic rather than an arbitrary one.
#[derive(Debug)]
pub struct ModelConfig {
    pub model_id: String,
    pub model_path: String,
    pub execution_provider: String,
    pub intra_op_threads: i32,
    pub fixed_batch_size: Option<usize>,
    pub vectorization_config: Option<VectorizationConfig>,
}

/// The JSON as written.
///
/// A separate struct because [`VectorizationConfig`] is not `Deserialize`: it holds
/// boxed trait objects and applies defaults as it is read, so it has its own
/// constructor. The nested value is handed to that constructor rather than being
/// modelled twice, which keeps one set of error messages for one shape of mistake.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelConfig {
    model_id: String,
    model_path: String,
    #[serde(default = "default_execution_provider")]
    execution_provider: String,
    #[serde(default = "default_intra_op_threads")]
    intra_op_threads: i32,
    fixed_batch_size: Option<usize>,
    vectorization_config: Option<serde_json::Value>,
}

impl ModelConfig {
    pub fn from_json(json: &str) -> Result<Self, crate::inference::InferenceError> {
        let raw: RawModelConfig = serde_json::from_str(json)?;

        // Zero rows is not a smaller batch, it is a model that can never be given
        // anything. Caught here because every later check would read it as "no pinning".
        if raw.fixed_batch_size == Some(0) {
            return Err(format!(
                "model {:?} declares \"fixed_batch_size\": 0, which no request could \
                 satisfy. Give the row count the model was exported for, or leave the \
                 field out to serve any number of rows.",
                raw.model_id
            )
            .into());
        }

        let vectorization_config = match raw.vectorization_config {
            Some(value) => Some(VectorizationConfig::from_json(&value.to_string()).map_err(
                |e| -> crate::inference::InferenceError {
                    format!("vectorization_config for model {:?}: {e}", raw.model_id).into()
                },
            )?),
            None => None,
        };

        Ok(ModelConfig {
            model_id: raw.model_id,
            model_path: raw.model_path,
            execution_provider: raw.execution_provider,
            intra_op_threads: raw.intra_op_threads,
            fixed_batch_size: raw.fixed_batch_size,
            vectorization_config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::vectorization_config::InputDataType;
    use crate::config::vectorization_config::{InputMode, VectorType};

    #[test]
    fn a_model_config_carries_the_model_and_its_hardware() {
        let config = ModelConfig::from_json(
            r#"{
                "model_id": "fraud-v3",
                "model_path": "s3://models/fraud/v3/model.onnx",
                "execution_provider": "coreml:cpu_and_neural_engine",
                "intra_op_threads": 4
            }"#,
        )
        .expect("valid");

        assert_eq!(config.model_id, "fraud-v3");
        assert_eq!(config.model_path, "s3://models/fraud/v3/model.onnx");
        assert_eq!(config.execution_provider, "coreml:cpu_and_neural_engine");
        assert_eq!(config.intra_op_threads, 4);
        assert!(
            config.vectorization_config.is_none(),
            "an absent sub-config means pass every input through"
        );
    }

    #[test]
    fn the_hardware_defaults_to_cpu_with_one_thread() {
        let config = ModelConfig::from_json(r#"{"model_id": "m", "model_path": "/tmp/m.onnx"}"#)
            .expect("valid");
        assert_eq!(config.execution_provider, "cpu");
        assert_eq!(config.intra_op_threads, 1);
        assert!(
            config.fixed_batch_size.is_none(),
            "dynamic batching is the default; pinning is opt-in"
        );
    }

    /// The declaration that lets a pinned-shape model be served. It is read here and
    /// enforced twice: against the model's own leading dimension when it loads, and
    /// against each request's row count before the engine sees it.
    #[test]
    fn a_declared_batch_size_is_carried_through() {
        let config = ModelConfig::from_json(
            r#"{
                "model_id": "m",
                "model_path": "/tmp/m.onnx",
                "execution_provider": "coreml",
                "fixed_batch_size": 1
            }"#,
        )
        .expect("valid");
        assert_eq!(config.fixed_batch_size, Some(1));
    }

    /// Zero rows is not a smaller batch, it is a model nothing can be sent to. Refused
    /// here because every later check would read it as "no pinning" and serve normally.
    #[test]
    fn a_zero_batch_size_is_refused_naming_the_field() {
        let err = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "fixed_batch_size": 0}"#,
        )
        .expect_err("no request can carry zero rows");
        assert!(
            err.to_string().contains("fixed_batch_size"),
            "should name the field: {err}"
        );
    }

    #[test]
    fn the_vectorization_sub_config_is_read_in_place() {
        let config = ModelConfig::from_json(
            r#"{
                "model_id": "m",
                "model_path": "/tmp/m.onnx",
                "vectorization_config": {
                    "data_type": "double",
                    "feature_transformations": {
                        "age": {"type": "identity", "default_val": [0.0]}
                    },
                    "feature_order": ["age"]
                }
            }"#,
        )
        .expect("valid");

        let vec_config = config.vectorization_config.expect("present");
        assert_eq!(
            vec_config.mode,
            InputMode::Vectorised {
                feature_order: vec!["age".into()],
                data_type: VectorType::Double,
            }
        );
    }

    #[test]
    fn a_bad_sub_config_names_the_model_and_the_reason() {
        let err = ModelConfig::from_json(
            r#"{
                "model_id": "fraud-v3",
                "model_path": "/tmp/m.onnx",
                "vectorization_config": {"feature_order": [], "model_inputs": {}}
            }"#,
        )
        .expect_err("two shapes cannot both be right");
        let m = err.to_string();
        assert!(
            m.contains("fraud-v3") && m.contains("feature_order"),
            "the error must locate the mistake: {m}"
        );
    }

    #[test]
    fn a_missing_required_field_is_refused() {
        assert!(
            ModelConfig::from_json(r#"{"model_path": "/tmp/m.onnx"}"#).is_err(),
            "model_id has no sensible default"
        );
        assert!(
            ModelConfig::from_json(r#"{"model_id": "m"}"#).is_err(),
            "model_path has no sensible default"
        );
    }

    #[test]
    fn a_misspelled_field_is_refused_rather_than_ignored() {
        let err = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "modelPath": "/other"}"#,
        )
        .expect_err("a typo must not be silently dropped");
        assert!(err.to_string().contains("modelPath"), "got {err}");
    }

    /// Every device spelling this config documents must actually parse, so a
    /// documented value cannot fail at startup.
    #[test]
    fn every_documented_provider_spelling_parses() {
        for spelling in [
            "cpu",
            "coreml",
            "coreml:all",
            "coreml:cpu_and_gpu",
            "coreml:cpu_and_neural_engine",
            "coreml:cpu_only",
            "xnnpack",
            "xnnpack:4",
            "tensorrt",
            "trt:0",
            "tensorrt:1",
            "migraphx",
            "migraphx:0",
        ] {
            let parsed: Result<onnxrt_rs::ExecutionProvider, _> = spelling.parse();
            assert!(
                parsed.is_ok(),
                "documented provider {spelling:?} does not parse: {:?}",
                parsed.err()
            );
        }
    }

    #[test]
    fn a_malformed_device_id_is_rejected() {
        let parsed: Result<onnxrt_rs::ExecutionProvider, _> = "tensorrt:gpu0".parse();
        assert!(parsed.is_err(), "a non-numeric device id must be rejected");
    }

    /// The lower-level vendor providers must be rejected with a pointer to the
    /// supported spelling, not silently accepted as an unknown provider name.
    #[test]
    fn cuda_and_rocm_point_at_the_supported_provider() {
        for (input, expected) in [("cuda", "tensorrt"), ("rocm", "migraphx")] {
            let err = input
                .parse::<onnxrt_rs::ExecutionProvider>()
                .expect_err("should be rejected");
            let message = err.to_string();
            assert!(
                message.contains(expected),
                "{input:?} should point at {expected:?}, got: {message}"
            );
        }
    }

    /// A complete configuration of each shape must parse, read from a file.
    ///
    /// The two fixtures in `test-data` are worked examples: between them they cover both
    /// shapes a `vectorization_config` can take, every kind of transformation, and every
    /// element type a named input can be. They are hand-written and belong to this test
    /// rather than being borrowed from `benchmark-data/generated`, whose files are
    /// generated -- a regenerated benchmark used to break this test, which told us
    /// nothing about the config shape.
    #[test]
    fn a_complete_configuration_of_each_shape_parses() {
        let read = |name: &str| {
            let path = std::path::Path::new("test-data").join(name);
            let json = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            ModelConfig::from_json(&json)
                .unwrap_or_else(|e| panic!("{} must parse: {e:?}", path.display()))
        };

        let vectorized = read("model_config_vectorized.json");
        assert_eq!(vectorized.model_id, "fixture-vectorized");
        let vectorization = vectorized
            .vectorization_config
            .expect("the vectorized fixture vectorises its features");
        assert!(
            matches!(
                vectorization.mode,
                InputMode::Vectorised {
                    data_type: VectorType::Float,
                    ..
                }
            ),
            "the vectorized fixture uses the single concatenated FP32 input"
        );
        // 3 embedding + 3 one-hot + 2 identity + 1 minmax + 1 standardization.
        assert_eq!(
            vectorization.vector_width().expect("a width"),
            10,
            "the width is the sum of the transformation widths, not a feature count"
        );

        let named = read("model_config_named.json");
        assert_eq!(named.model_id, "fixture-named");
        assert_eq!(named.execution_provider, "coreml:cpu_and_neural_engine");
        assert_eq!(named.intra_op_threads, 2);
        let InputMode::Named { model_inputs } = named
            .vectorization_config
            .expect("the named fixture declares its inputs")
            .mode
        else {
            panic!("the named fixture declares one input per feature");
        };
        assert_eq!(
            model_inputs
                .iter()
                .map(|(name, kind)| (name.as_str(), *kind))
                .collect::<Vec<_>>(),
            vec![
                ("age", InputDataType::Float),
                ("city", InputDataType::FloatArray),
                ("tags", InputDataType::String),
                ("token_id", InputDataType::Long),
            ],
            "declared inputs are read in name order, whatever order the JSON used"
        );
    }
}
