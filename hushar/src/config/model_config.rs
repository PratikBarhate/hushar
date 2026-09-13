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
/// One model, and how to serve it.
///
/// Fields:
/// - `model_id` — recorded on every inference log row, so logs join to a model version.
/// - `model_path` — where the model file is, carrying its own URI scheme.
/// - `execution_provider` — the hardware backend. See below.
/// - `mini_batch_size` — split a request's rows into batches of this many, scored at
///   the same time. See below.
/// - `is_fixed` — pad the last mini batch up to `mini_batch_size`. See below.
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
/// **Why it lowers latency.** Threads inside one operator stop helping well past a
/// point -- the operators are a chain and each one is a barrier -- so a big batch has a
/// latency floor no thread count clears. Separate mini-batches share no barrier at all,
/// so `n` of them on `n` cores approach the cost of *one*. Splitting 100 rows into 20
/// therefore costs about what 5 rows cost, not a twentieth of 100 rows' wall clock.
/// It buys latency with a little more total CPU: each batch repeats the per-call
/// overhead, and small batches are less efficient per row.
///
/// Pair it with `threading.intra_op_threads: 1`. The parallelism now comes from running
/// mini-batches together, and a pool inside each one competes with that for the same
/// cores.
///
/// **`is_fixed` is what a pinned graph needs.** A model exported with its leading
/// dimension fixed can only be given batches of that one size, so it needs
/// `mini_batch_size` equal to that number and `is_fixed: true`; the padded rows are
/// scored and discarded. A pinned axis disagreeing with the declared size, or declared
/// without `is_fixed`, is refused when the model loads rather than failing every
/// request inside the engine.
///
/// A padding row carries no features, so every input takes the same path as a request
/// that omitted that feature: its transformation's `default_val`, or zeros and empty
/// strings for inputs passed through verbatim. Nothing new has to be correct for
/// padding to be correct.
///
/// **It never constrains callers.** A request of any size is served either way.
#[derive(Debug)]
pub struct ModelConfig {
    pub model_id: String,
    pub model_path: String,
    pub execution_provider: String,
    pub mini_batch_size: Option<usize>,
    pub is_fixed: bool,
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
    mini_batch_size: Option<usize>,
    #[serde(default)]
    is_fixed: bool,
    vectorization_config: Option<serde_json::Value>,
}

impl ModelConfig {
    pub fn from_json(json: &str) -> Result<Self, crate::inference::InferenceError> {
        let raw: RawModelConfig = serde_json::from_str(json)?;

        // Zero rows is not a smaller batch, it is a model that can never be given
        // anything. Caught here because every later check would read it as "no splitting".
        if raw.mini_batch_size == Some(0) {
            return Err(format!(
                "model {:?} declares \"mini_batch_size\": 0, which no request could \
                 satisfy. Give the rows per mini batch, or leave the field out to score \
                 each request as one batch.",
                raw.model_id
            )
            .into());
        }

        // `is_fixed` only means anything alongside a size, and a config setting it alone
        // has almost certainly lost the size rather than meant nothing by it.
        if raw.is_fixed && raw.mini_batch_size.is_none() {
            return Err(format!(
                "model {:?} declares \"is_fixed\": true without \"mini_batch_size\", so \
                 there is no size to pad to. Give the row count the model's graph is \
                 pinned to, or drop \"is_fixed\".",
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
            mini_batch_size: raw.mini_batch_size,
            is_fixed: raw.is_fixed,
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
                "mini_batch_size": 4
            }"#,
        )
        .expect("valid");

        assert_eq!(config.model_id, "fraud-v3");
        assert_eq!(config.model_path, "s3://models/fraud/v3/model.onnx");
        assert_eq!(config.execution_provider, "coreml:cpu_and_neural_engine");
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
        assert!(
            config.mini_batch_size.is_none(),
            "one batch per request is the default; splitting is opt-in"
        );
        assert!(!config.is_fixed, "padding is opt-in with the size");
    }

    /// The declaration that lets a pinned-shape model be served: the size plus the flag
    /// that says the last batch is padded to it. Read here and enforced against the
    /// model's own leading dimension when it loads.
    #[test]
    fn a_declared_batch_size_is_carried_through() {
        let config = ModelConfig::from_json(
            r#"{
                "model_id": "m",
                "model_path": "/tmp/m.onnx",
                "execution_provider": "coreml",
                "mini_batch_size": 1,
                "is_fixed": true
            }"#,
        )
        .expect("valid");
        assert_eq!(config.mini_batch_size, Some(1));
        assert!(config.is_fixed);
    }

    /// The latency case: a size without the flag, which splits and leaves the last batch
    /// short. Separate from the pinned case because the two differ only in this flag and
    /// a reader should see both spellings.
    #[test]
    fn a_size_without_the_flag_splits_without_padding() {
        let config = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "mini_batch_size": 10}"#,
        )
        .expect("valid");
        assert_eq!(config.mini_batch_size, Some(10));
        assert!(
            !config.is_fixed,
            "a short last batch is the default, so a dynamic model needs no extra field"
        );
    }

    /// A flag with nothing to pad to. Refused naming both fields, because the config has
    /// almost certainly lost the size rather than meant nothing by the flag.
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

    /// Zero rows is not a smaller batch, it is a model nothing can be sent to. Refused
    /// here because every later check would read it as "no pinning" and serve normally.
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
