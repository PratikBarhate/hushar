// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Everything about one model: where it is, how it runs, and how features reach it.
//!
//! This file describes the model and the hardware it runs on; the service configuration
//! describes the process, so `execution_provider` lives here.
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
//! `vectorization_config` is required, and it is one of two shapes: `feature_order`
//! concatenates every feature into the model's one input, and `model_inputs` declares one
//! input per feature and feeds each verbatim.

use crate::config::vectorization_config::VectorizationConfig;
use serde::Deserialize;

/// The default execution provider, `cpu`.
fn default_execution_provider() -> String {
    "cpu".to_owned()
}

/// One model, and how to serve it.
///
/// # Fields
///
/// * `model_id` - recorded on every inference log row, so logs join to a model version
/// * `model_path` - where the model file is, carrying its own URI scheme
/// * `execution_provider` - the hardware backend, defaulting to `cpu`; see below
/// * `mini_batch_size` - split a request's rows into batches of this many, scored at the
///   same time; optional, and must be greater than zero; see below
/// * `is_fixed` - pad the last mini batch up to `mini_batch_size`; defaults to false and
///   requires `mini_batch_size`; see below
/// * `vectorization_config` - how features become model inputs; required, see below
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
/// Splitting trades a little more total CPU for lower latency: each batch repeats the
/// per-call overhead, and small batches are less efficient per row. Pair it with
/// `threading.intra_op_threads: 1`, since the parallelism then comes from running mini
/// batches together and a pool inside each one competes for the same cores.
///
/// **`is_fixed` is what a pinned graph needs.** A model exported with its leading
/// dimension fixed can only be given batches of that one size, so it needs
/// `mini_batch_size` equal to that number and `is_fixed: true`; the padded rows are
/// scored and discarded. A pinned axis disagreeing with the declared size, or declared
/// without `is_fixed`, is refused when the model loads.
///
/// A padding row carries no features, so every input takes the same path as a request
/// that omitted that feature: its transformation's `default_val`, or zeros and empty
/// strings for inputs passed through verbatim.
///
/// **It never constrains callers.** A request of any size is served either way.
#[derive(Debug)]
pub struct ModelConfig {
    pub model_id: String,
    pub model_path: String,
    pub execution_provider: String,
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
    #[serde(default = "default_execution_provider")]
    execution_provider: String,
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
                 features reach it. Declare \"model_inputs\" to feed one input per \
                 feature, or \"feature_order\" to concatenate them into one.",
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
                "mini_batch_size": 4,
                "vectorization_config": {"feature_order": ["a"]}
            }"#,
        )
        .expect("valid");

        assert_eq!(config.model_id, "fraud-v3");
        assert_eq!(config.model_path, "s3://models/fraud/v3/model.onnx");
        assert_eq!(config.execution_provider, "coreml:cpu_and_neural_engine");
    }

    #[test]
    fn a_model_without_a_vectorization_config_is_refused() {
        let err = ModelConfig::from_json(r#"{"model_id": "m", "model_path": "/tmp/m.onnx"}"#)
            .expect_err("nothing says how features reach the model");
        let m = err.to_string();
        assert!(
            m.contains("model_inputs") && m.contains("feature_order"),
            "the refusal should name both shapes; got {m}"
        );
    }

    #[test]
    fn the_hardware_defaults_to_cpu_with_one_thread() {
        let config = ModelConfig::from_json(
            r#"{"model_id": "m", "model_path": "/tmp/m.onnx", "vectorization_config": {"feature_order": ["a"]}}"#,
        )
        .expect("valid");
        assert_eq!(config.execution_provider, "cpu");
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
        let InputMode::Vectorised {
            feature_order,
            data_type,
            ..
        } = &vec_config.mode
        else {
            panic!("feature_order must select the vectorised shape");
        };
        assert_eq!(feature_order.as_slice(), ["age"]);
        assert_eq!(*data_type, VectorType::Double);
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

    /// Asserts the `test-data` fixture of each shape parses, read from a file.
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
        let vectorization = vectorized.vectorization_config;
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
        assert_eq!(
            vectorization.vector_width().expect("a width"),
            10,
            "the width is the sum of the transformation widths, not a feature count"
        );

        let named = read("model_config_named.json");
        assert_eq!(named.model_id, "fixture-named");
        assert_eq!(named.execution_provider, "coreml:cpu_and_neural_engine");
        let InputMode::Named { model_inputs } = named.vectorization_config.mode else {
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
