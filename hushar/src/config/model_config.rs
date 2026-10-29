// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Everything about one model: where it is, how it runs, and how features reach it.
//!
//! This file describes the model and the hardware it runs on. Existing ONNX Runtime
//! configurations may keep the top-level `execution_provider`; new configurations can
//! use an explicit backend object, which is required for Neuron.
//!
//! ```json
//! {
//!   "model_id": "fraud-v3",
//!   "model_path": "s3://models/fraud/v3/model.neff",
//!   "backend": {
//!     "type": "neuron",
//!     "start_neuron_core": 0,
//!     "neuron_cores_per_replica": 1,
//!     "execution_contexts_per_replica": 2
//!   },
//!   "vectorization_config": {
//!     "data_type": "float",
//!     "feature_transformations": { "age": { "type": "identity", "default_val": [0.0] } },
//!     "feature_order": ["age", "city"]
//!   }
//! }
//! ```
//!
//! `vectorization_config` is required: `feature_order` names the features that are
//! transformed and concatenated into the model's one input.

use crate::config::vectorization_config::VectorizationConfig;
use serde::Deserialize;

/// The default execution provider, `cpu`.
fn default_execution_provider() -> String {
    "cpu".to_owned()
}

fn default_neuron_core_count() -> i32 {
    1
}

fn default_execution_contexts() -> usize {
    1
}

fn default_replicas_per_core() -> usize {
    1
}

/// ONNX Runtime settings that belong to one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxRuntimeConfig {
    pub execution_provider: String,
}

/// AWS Neuron placement and execution settings that belong to one compiled model.
///
/// Replicas are controlled by `threading.sessions_per_model`. Each logical replica uses
/// `neuron_cores_per_replica` consecutive cores, beginning at `start_neuron_core`, and
/// loads one complete NEFF copy on each core. This is data parallelism across a request's
/// mini batches, not model sharding. `execution_contexts_per_replica` owns the reusable
/// tensor sets and distributes them round-robin across those per-core copies; it must be
/// at least the core count.
///
/// `replicas_per_neuron_core` loads that many logical replicas onto the same core range
/// before moving to the next one. Above one, each core holds another copy of the weights
/// per replica and can overlap their host-side work, though device execution still
/// serializes on that core.
///
/// A request leases one logical replica while it scores, and each of its
/// `ceil(rows / mini_batch_size)` fixed executions leases one tensor set from it. After
/// providing at least one context per core, the useful context count is therefore the
/// executions one request produces: fewer serialize pieces, more are never leased.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuronConfig {
    pub start_neuron_core: i32,
    pub neuron_cores_per_replica: i32,
    pub replicas_per_neuron_core: usize,
    pub execution_contexts_per_replica: usize,
}

/// The engine and engine-specific model settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendConfig {
    OnnxRuntime(OnnxRuntimeConfig),
    Neuron(NeuronConfig),
}

impl BackendConfig {
    /// A concise engine name for startup errors and benchmark metadata.
    pub fn name(&self) -> &'static str {
        match self {
            Self::OnnxRuntime(_) => "onnxruntime",
            Self::Neuron(_) => "neuron",
        }
    }

    /// The ONNX Runtime provider, when this is an ONNX Runtime model.
    #[cfg(test)]
    pub fn execution_provider(&self) -> Option<&str> {
        match self {
            Self::OnnxRuntime(config) => Some(&config.execution_provider),
            Self::Neuron(_) => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RawBackendConfig {
    #[serde(rename = "onnxruntime")]
    OnnxRuntime {
        #[serde(default = "default_execution_provider")]
        execution_provider: String,
    },
    Neuron {
        #[serde(default)]
        start_neuron_core: i32,
        #[serde(default = "default_neuron_core_count")]
        neuron_cores_per_replica: i32,
        #[serde(default = "default_replicas_per_core")]
        replicas_per_neuron_core: usize,
        #[serde(default = "default_execution_contexts")]
        execution_contexts_per_replica: usize,
    },
}

/// One model, and how to serve it.
///
/// # Fields
///
/// * `model_id` - recorded on every inference log row, so logs join to a model version
/// * `model_path` - where the model file is, carrying its own URI scheme
/// * `backend` - the engine and only its own settings. ONNX Runtime selects an execution
///   provider; Neuron selects core placement and reusable execution contexts
/// * `mini_batch_size` - split a request's rows into batches of this many, scored at the
///   same time; optional, and must be greater than zero; see below
/// * `is_fixed` - pad the last mini batch up to `mini_batch_size`; defaults to false and
///   requires `mini_batch_size`; see below
/// * `vectorization_config` - how features become model inputs; required, see below
///
/// # Backends
///
/// Leaving out `backend` preserves the original ONNX Runtime form and reads the
/// top-level `execution_provider`, defaulting to `cpu`. The explicit equivalent is
/// `{"type":"onnxruntime","execution_provider":"cpu"}`. A Neuron backend requires
/// a compiled NEFF and fixed mini-batch configuration. `threading.sessions_per_model`
/// loads consecutive replicas; NeuronCore placement and tensor-context concurrency stay
/// in the backend object and are never passed to ONNX Runtime.
///
/// # Execution providers
///
/// Accepts `cpu`, `coreml`, `coreml:cpu_and_neural_engine`, `xnnpack`,
/// `xnnpack:<threads>`, `tensorrt[:device]` for NVIDIA and `migraphx[:device]` for AMD.
/// Any other value is passed to ONNX Runtime as a provider name, so providers this
/// service does not model can still be used.
///
/// `cuda` and `rocm` are rejected: use `tensorrt` or `migraphx`, which the error names.
///
/// The provider must be compiled into the `libonnxruntime` on the host; startup fails
/// with the list of available providers if it is not.
///
/// # Mini batches
///
/// `mini_batch_size` cuts a request's rows into batches of that many, **scored at the
/// same time** rather than one after another. `is_fixed` says whether the last batch is
/// padded up to the size or left short.
///
/// Left out, a request is one batch of whatever size it arrived as, and the model's row
/// axis must be dynamic. That stays the default.
///
/// ```text
/// mini_batch_size: 4, request of 6 rows
///
///   is_fixed: false            is_fixed: true
///   r0 r1 r2 r3 │ r4 r5        r0 r1 r2 r3 │ r4 r5 ·· ··   ·· = padding row
///   └── run ──┘   └─ run ─┘    └── run ──┘   └─── run ───┘
///       both at the same time      both at the same time
///            ▼                            ▼
///   s0 s1 s2 s3   s4 s5        s0 s1 s2 s3   s4 s5 xx xx   xx = scored, discarded
///   └───── 6 rows out ─┘       └────── 6 rows out ───┘
/// ```
///
/// Splitting trades more total work for lower latency because each batch repeats per-call
/// overhead. With ONNX Runtime, pair it with `threading.intra_op_threads: 1`; with Neuron,
/// `backend.execution_contexts_per_replica` bounds how many pieces run concurrently.
///
/// **`is_fixed` is what a pinned graph needs.** A model exported with its leading
/// dimension fixed can only be given batches of that one size, so it needs
/// `mini_batch_size` equal to that number and `is_fixed: true`; the padded rows are
/// scored and discarded. A pinned axis disagreeing with the declared size, or declared
/// without `is_fixed`, is refused when the model loads.
///
/// A padding row carries no features, so it takes the same path as a request that omitted
/// them: every transformation contributes its `default_val`.
///
/// **It never constrains callers.** A request of any size is served either way.
#[derive(Debug)]
pub struct ModelConfig {
    pub model_id: String,
    pub model_path: String,
    pub backend: BackendConfig,
    pub mini_batch_size: Option<usize>,
    pub is_fixed: bool,
    pub vectorization_config: VectorizationConfig,
}

/// The JSON as written.
///
/// A separate struct because [`VectorizationConfig`] is not `Deserialize`: it holds
/// boxed trait objects and applies defaults as it is read, so it has its own
/// constructor.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelConfig {
    model_id: String,
    model_path: String,
    execution_provider: Option<String>,
    backend: Option<RawBackendConfig>,
    mini_batch_size: Option<usize>,
    #[serde(default)]
    is_fixed: bool,
    vectorization_config: Option<serde_json::Value>,
}

impl ModelConfig {
    /// Reads a model configuration from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON does not parse, if `mini_batch_size` is zero, if
    /// `is_fixed` is set without a `mini_batch_size` to pad to, or if
    /// `vectorization_config` is missing or invalid.
    pub fn from_json(json: &str) -> Result<Self, crate::inference::InferenceError> {
        let raw: RawModelConfig = serde_json::from_str(json)?;

        let backend = match (raw.backend, raw.execution_provider) {
            (None, execution_provider) => BackendConfig::OnnxRuntime(OnnxRuntimeConfig {
                execution_provider: execution_provider.unwrap_or_else(default_execution_provider),
            }),
            (Some(_), Some(_)) => {
                return Err(format!(
                    "model {:?} sets both \"backend\" and the legacy \"execution_provider\"; put the provider inside an onnxruntime backend, or remove it for Neuron",
                    raw.model_id
                )
                .into());
            }
            (Some(RawBackendConfig::OnnxRuntime { execution_provider }), None) => {
                BackendConfig::OnnxRuntime(OnnxRuntimeConfig { execution_provider })
            }
            (
                Some(RawBackendConfig::Neuron {
                    start_neuron_core,
                    neuron_cores_per_replica,
                    replicas_per_neuron_core,
                    execution_contexts_per_replica,
                }),
                None,
            ) => {
                if start_neuron_core < 0 {
                    return Err(format!(
                        "model {:?} sets backend.start_neuron_core to {start_neuron_core}; NeuronCore indexes start at 0",
                        raw.model_id
                    )
                    .into());
                }
                if neuron_cores_per_replica <= 0 {
                    return Err(format!(
                        "model {:?} sets backend.neuron_cores_per_replica to {neuron_cores_per_replica}; each replica needs at least one NeuronCore",
                        raw.model_id
                    )
                    .into());
                }
                if replicas_per_neuron_core == 0 {
                    return Err(format!(
                        "model {:?} sets backend.replicas_per_neuron_core to 0; each core range holds at least one replica",
                        raw.model_id
                    )
                    .into());
                }
                if execution_contexts_per_replica == 0 {
                    return Err(format!(
                        "model {:?} sets backend.execution_contexts_per_replica to 0; each replica needs at least one reusable tensor set",
                        raw.model_id
                    )
                    .into());
                }
                BackendConfig::Neuron(NeuronConfig {
                    start_neuron_core,
                    neuron_cores_per_replica,
                    replicas_per_neuron_core,
                    execution_contexts_per_replica,
                })
            }
        };

        if raw.mini_batch_size == Some(0) {
            return Err(format!(
                "model {:?} declares \"mini_batch_size\": 0, which no request could \
                 satisfy. Give the rows per mini batch, or leave the field out to score \
                 each request as one batch.",
                raw.model_id
            )
            .into());
        }

        if raw.is_fixed && raw.mini_batch_size.is_none() {
            return Err(format!(
                "model {:?} declares \"is_fixed\": true without \"mini_batch_size\", so \
                 there is no size to pad to. Give the row count the model's graph is \
                 pinned to, or drop \"is_fixed\".",
                raw.model_id
            )
            .into());
        }

        let Some(value) = raw.vectorization_config else {
            return Err(format!(
                "model {:?} has no \"vectorization_config\", so nothing says how its \
                 features reach it. Declare \"feature_order\" to name the features \
                 concatenated into the model's input.",
                raw.model_id
            )
            .into());
        };
        let vectorization_config = VectorizationConfig::from_json(&value.to_string()).map_err(
            |e| -> crate::inference::InferenceError {
                format!("vectorization_config for model {:?}: {e}", raw.model_id).into()
            },
        )?;

        Ok(ModelConfig {
            model_id: raw.model_id,
            model_path: raw.model_path,
            backend,
            mini_batch_size: raw.mini_batch_size,
            is_fixed: raw.is_fixed,
            vectorization_config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::vectorization_config::VectorType;

    #[test]
    fn a_model_config_carries_the_model_and_its_hardware() {
        let config = ModelConfig::from_json(
            r#"{
                "model_id": "fraud-v3",
                "model_path": "s3://models/fraud/v3/model.onnx",
                "execution_provider": "coreml:cpu_and_neural_engine",
                "mini_batch_size": 4,
                "vectorization_config": {"feature_order": ["a"]}
            }"#,
        )
        .expect("valid");

        assert_eq!(config.model_id, "fraud-v3");
        assert_eq!(config.model_path, "s3://models/fraud/v3/model.onnx");
        assert_eq!(
            config.backend.execution_provider(),
            Some("coreml:cpu_and_neural_engine")
        );
    }

    #[test]
    fn a_model_without_a_vectorization_config_is_refused() {
        let err = ModelConfig::from_json(r#"{"model_id": "m", "model_path": "/tmp/m.onnx"}"#)
            .expect_err("nothing says how features reach the model");
        assert!(
            err.to_string().contains("feature_order"),
            "the refusal should name the field; got {err}"
        );
    }

    #[test]
    fn the_hardware_defaults_to_cpu_with_one_thread() {
        let config = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "vectorization_config": {"feature_order": ["a"]}}"#,
        )
        .expect("valid");
        assert_eq!(config.backend.execution_provider(), Some("cpu"));
        assert!(
            config.mini_batch_size.is_none(),
            "one batch per request is the default; splitting is opt-in"
        );
        assert!(!config.is_fixed, "padding is opt-in with the size");
    }

    /// Asserts a declared `mini_batch_size` and `is_fixed` are carried through.
    #[test]
    fn a_declared_batch_size_is_carried_through() {
        let config = ModelConfig::from_json(
            r#"{
                "model_id": "m",
                "model_path": "/tmp/m.onnx",
                "execution_provider": "coreml",
                "mini_batch_size": 1,
                "is_fixed": true,
                "vectorization_config": {"feature_order": ["a"]}
            }"#,
        )
        .expect("valid");
        assert_eq!(config.mini_batch_size, Some(1));
        assert!(config.is_fixed);
    }

    /// Asserts a size without `is_fixed` splits and leaves the last batch short.
    #[test]
    fn a_size_without_the_flag_splits_without_padding() {
        let config = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "mini_batch_size": 10, "vectorization_config": {"feature_order": ["a"]}}"#,
        )
        .expect("valid");
        assert_eq!(config.mini_batch_size, Some(10));
        assert!(
            !config.is_fixed,
            "a short last batch is the default, so a dynamic model needs no extra field"
        );
    }

    /// Asserts `is_fixed` without `mini_batch_size` is refused, naming both fields.
    #[test]
    fn the_flag_without_a_size_is_refused() {
        let err = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "is_fixed": true}"#,
        )
        .expect_err("a size is required to pad to");
        let m = err.to_string();
        assert!(
            m.contains("is_fixed") && m.contains("mini_batch_size"),
            "the message should name both fields, got: {m}"
        );
    }

    /// Asserts a `mini_batch_size` of zero is refused, naming the field.
    #[test]
    fn a_zero_batch_size_is_refused_naming_the_field() {
        let err = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "mini_batch_size": 0}"#,
        )
        .expect_err("no request can carry zero rows");
        assert!(
            err.to_string().contains("mini_batch_size"),
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

        let vec_config = config.vectorization_config;
        assert_eq!(vec_config.feature_order, ["age"]);
        assert_eq!(vec_config.data_type, VectorType::Double);
    }

    #[test]
    fn a_bad_sub_config_names_the_model_and_the_reason() {
        let err = ModelConfig::from_json(
            r#"{
                "model_id": "fraud-v3",
                "model_path": "/tmp/m.onnx",
                "vectorization_config": {"feature_order": []}
            }"#,
        )
        .expect_err("an empty order concatenates nothing");
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

    /// Asserts every provider spelling this config documents parses.
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

    /// Asserts `cuda` and `rocm` are rejected with the supported provider named.
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

    /// Asserts the complete `test-data` fixture parses, read from a file.
    #[test]
    fn a_complete_configuration_parses() {
        let path = std::path::Path::new("test-data").join("model_config_vectorized.json");
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let config = ModelConfig::from_json(&json)
            .unwrap_or_else(|e| panic!("{} must parse: {e:?}", path.display()));

        assert_eq!(config.model_id, "fixture-vectorized");
        let vectorization = config.vectorization_config;
        assert_eq!(
            vectorization.data_type,
            VectorType::Float,
            "the fixture feeds one concatenated FP32 input"
        );
        assert_eq!(
            vectorization.vector_width().expect("a width"),
            10,
            "the width is the sum of the transformation widths, not a feature count"
        );
    }

    #[test]
    fn a_neuron_backend_carries_only_neuron_settings() {
        let config = ModelConfig::from_json(
            r#"{
                "model_id": "compiled",
                "model_path": "/models/compiled.neff",
                "backend": {
                    "type": "neuron",
                    "start_neuron_core": 1,
                    "neuron_cores_per_replica": 2,
                    "execution_contexts_per_replica": 4
                },
                "mini_batch_size": 8,
                "is_fixed": true,
                "vectorization_config": {"feature_order": ["features"]}
            }"#,
        )
        .expect("valid");

        assert_eq!(
            config.backend,
            BackendConfig::Neuron(NeuronConfig {
                start_neuron_core: 1,
                neuron_cores_per_replica: 2,
                replicas_per_neuron_core: 1,
                execution_contexts_per_replica: 4,
            })
        );
        assert_eq!(config.backend.execution_provider(), None);
    }

    /// Replicas may share a core range when asked, and a zero count is refused.
    #[test]
    fn replicas_per_core_is_opt_in_and_positive() {
        let json = |count: usize| {
            format!(
                r#"{{
                    "model_id": "shared",
                    "model_path": "/models/compiled.neff",
                    "backend": {{"type": "neuron", "replicas_per_neuron_core": {count}}},
                    "mini_batch_size": 8,
                    "is_fixed": true,
                    "vectorization_config": {{"feature_order": ["features"]}}
                }}"#
            )
        };
        let shared = ModelConfig::from_json(&json(2)).expect("valid");
        let BackendConfig::Neuron(neuron) = shared.backend else {
            panic!("a Neuron backend was declared");
        };
        assert_eq!(neuron.replicas_per_neuron_core, 2);
        assert_eq!(
            neuron.neuron_cores_per_replica, 1,
            "the other placement defaults hold"
        );

        let error = ModelConfig::from_json(&json(0)).expect_err("zero replicas per core");
        assert!(
            error.to_string().contains("replicas_per_neuron_core"),
            "got {error}"
        );
    }

    #[test]
    fn backend_specific_settings_cannot_be_mixed() {
        let mixed = r#"{
            "model_id": "mixed",
            "model_path": "/models/compiled.neff",
            "execution_provider": "cpu",
            "backend": {"type": "neuron"},
            "vectorization_config": {"feature_order": ["features"]}
        }"#;
        let error = ModelConfig::from_json(mixed).expect_err("two engines were configured");
        assert!(
            error.to_string().contains("both \"backend\"")
                && error.to_string().contains("execution_provider"),
            "{error}"
        );
    }

    #[test]
    fn invalid_neuron_placement_is_refused() {
        for backend in [
            r#"{"type":"neuron","start_neuron_core":-1}"#,
            r#"{"type":"neuron","neuron_cores_per_replica":0}"#,
            r#"{"type":"neuron","execution_contexts_per_replica":0}"#,
        ] {
            let json = format!(
                r#"{{"model_id":"m","model_path":"m.neff","backend":{backend},"vectorization_config":{{"feature_order":["features"]}}}}"#
            );
            assert!(ModelConfig::from_json(&json).is_err(), "accepted {backend}");
        }
    }

    /// Asserts a model config of this backend spelling parses, whatever the hardware.
    ///
    /// Every feature is transformed into floats on the way in, so no element type a
    /// backend lacks can reach the model.
    #[test]
    fn a_concatenated_float_input_reaches_every_backend() {
        for backend in [
            r#""backend":{"type":"neuron"}"#,
            r#""execution_provider":"cpu""#,
            r#""execution_provider":"xnnpack""#,
            r#""execution_provider":"tensorrt""#,
            r#""execution_provider":"migraphx:0""#,
            r#""execution_provider":"coreml:cpu_only""#,
        ] {
            let json = format!(
                r#"{{"model_id":"m","model_path":"/models/m",{backend},
                   "vectorization_config":{{"feature_order":["city"]}}}}"#
            );
            ModelConfig::from_json(&json)
                .unwrap_or_else(|e| panic!("{backend} takes a float vector: {e}"));
        }
    }
}
