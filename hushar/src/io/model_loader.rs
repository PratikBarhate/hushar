// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Builds the inference backend from a model and the service configuration.

use std::sync::Arc;

use onnxrt_rs::ExecutionProvider;

use crate::inference::{InferenceBackend, OnnxRuntimeBackend};

/// Loads an ONNX model onto the configured execution provider.
///
/// Returns the backend behind an [`Arc`] so every request thread shares one
/// model: ONNX Runtime supports concurrent `Run` on a single session, so no
/// per-thread copy is needed.
///
/// # Arguments
///
/// * `model_bytes` - the ONNX model protobuf
/// * `execution_provider` - the hardware backend, as spelled in the service
///   config; see [`crate::config::HusharServiceConfig::execution_provider`]
/// * `intra_op_threads` - threads ONNX Runtime may use within one operator
/// * `session_options` - ONNX Runtime's string-keyed session settings, such as
///   thread affinity and spinning
/// * `mini_batch` - how this deployment cuts a request's rows, when the model
///   configuration says to. A model with a pinned leading dimension loads only when a
///   padded size agrees with it.
///
/// # Errors
///
/// Fails if the provider string cannot be parsed, if the ONNX Runtime library is
/// missing, if the requested provider is not compiled into it, or if the model
/// does not parse or declares no concrete feature width.
pub fn load_onnx_model(
    model_bytes: &[u8],
    execution_provider: &str,
    intra_op_threads: i32,
    session_options: &[(String, String)],
    mini_batch: Option<crate::inference::scoring::MiniBatch>,
) -> Result<Arc<dyn InferenceBackend>, crate::inference::InferenceError> {
    let provider: ExecutionProvider = execution_provider.parse()?;
    let threads = (intra_op_threads > 0).then_some(intra_op_threads);
    let backend =
        OnnxRuntimeBackend::load(model_bytes, &provider, threads, mini_batch, session_options)?;
    Ok(Arc::new(backend))
}

/// Loads the same model into one session per entry in `session_options`.
///
/// One session per pool, rather than one session shared by every caller. The intra-op pool
/// belongs to the session, so this is the only way to give concurrent requests pools that
/// do not contend — and each entry carries its own
/// `session.intra_op_thread_affinities`, which is what keeps a pool on its own cores.
///
/// Every session reads the same bytes, so the weights are duplicated: memory is
/// `sessions x model size`. The graph and its signature are identical by construction,
/// which is what lets the caller describe the set from any one of them.
///
/// # Errors
///
/// As [`load_onnx_model`]. The first failure is returned, naming the session that hit it,
/// since a partially loaded set is of no use.
pub fn load_onnx_sessions(
    model_bytes: &[u8],
    execution_provider: &str,
    intra_op_threads: i32,
    session_options: &[Vec<(String, String)>],
    mini_batch: Option<crate::inference::scoring::MiniBatch>,
) -> Result<Vec<Arc<dyn InferenceBackend>>, crate::inference::InferenceError> {
    session_options
        .iter()
        .enumerate()
        .map(|(index, options)| {
            load_onnx_model(
                model_bytes,
                execution_provider,
                intra_op_threads,
                options,
                mini_batch,
            )
            .map_err(|e| {
                crate::inference::InferenceError::from(format!(
                    "session {} of {}: {e}",
                    index + 1,
                    session_options.len()
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// The fixture the service's own tests use: 3 features in, 2 sigmoid outputs.
    fn test_model() -> Option<Vec<u8>> {
        let path = Path::new("test-data/sigmoid_model_3.onnx");
        match std::fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                eprintln!("skipping: cannot read {}: {e}", path.display());
                None
            }
        }
    }

    /// Whether an ONNX Runtime library is present. These tests need one, and skip
    /// when it is absent so the suite stays green on machines without it.
    fn runtime_available() -> bool {
        match onnxrt_rs::Api::load() {
            Ok(_) => true,
            Err(e) => {
                eprintln!("skipping: no ONNX Runtime available ({e})");
                false
            }
        }
    }

    /// One row of the fixture's features.
    ///
    /// Only the accelerator modules use this, so a build with no accelerator
    /// feature leaves it unused.
    #[allow(dead_code)]
    fn single_row() -> Vec<f32> {
        vec![1.0, 2.0, 3.0]
    }

    /// Runs `values` as `rows` rows through the backend's single input and returns
    /// the primary output, one `Vec` per row.
    ///
    /// The input's name, element type and shape all come from the model, so this
    /// helper works for any single-input model rather than the fixture only.
    fn run_rows(
        backend: &std::sync::Arc<dyn InferenceBackend>,
        values: Vec<f32>,
        rows: usize,
    ) -> Result<Vec<Vec<f32>>, crate::inference::InferenceError> {
        use crate::inference::batch::{FeatureData, InputBatch};

        let spec = backend
            .inputs()
            .first()
            .ok_or("model declares no inputs")?
            .clone();
        let mut batch = InputBatch::new(rows);
        batch.push(spec.name.clone(), FeatureData::F32(values))?;
        let produced = backend.run(batch)?;
        let primary = produced.first().ok_or("model returned no outputs")?;

        let mut out = Vec::with_capacity(produced.rows());
        for i in 0..produced.rows() {
            out.push(
                primary
                    .row_as_f32(i)
                    .ok_or_else(|| format!("no values for row {i}"))?,
            );
        }
        Ok(out)
    }

    /// Scores for one row on `provider`, or `None` when the accelerator is absent
    /// from the loaded library.
    ///
    /// Distinguishes "this build was not compiled for it" and "the library does
    /// not have it" -- both legitimate reasons to skip -- from a genuine failure,
    /// which panics.
    ///
    /// Only the accelerator modules use this, so a build with no accelerator
    /// feature leaves it unused.
    #[allow(dead_code)]
    fn scores_on(model: &[u8], provider: &str) -> Option<Vec<f32>> {
        match load_onnx_model(model, provider, 1, &[], None) {
            Ok(backend) => {
                let rows = run_rows(&backend, single_row(), 1)
                    .unwrap_or_else(|e| panic!("{provider} inference failed: {e}"));
                Some(rows.into_iter().next().expect("one row"))
            }
            Err(e) => {
                let message = e.to_string();
                if message.contains("not available") || message.contains("does not enable") {
                    eprintln!("skipping {provider}: {message}");
                    None
                } else {
                    panic!("{provider} failed unexpectedly: {message}");
                }
            }
        }
    }

    /// Asserts an accelerator agrees with CPU, within a tolerance that allows for
    /// different kernels and accumulation orders but not a genuinely wrong answer.
    ///
    /// Only the accelerator modules use this, so a CPU-only build leaves it unused.
    #[allow(dead_code)]
    fn assert_matches_cpu(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: wrong score count");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-4,
                "{label} score {i}: got {a}, CPU gave {e}"
            );
        }
    }

    fn assert_sigmoid_range(label: &str, scores: &[f32]) {
        assert_eq!(scores.len(), 2, "{label}: fixture emits 2 scores");
        for s in scores {
            assert!(*s > 0.0 && *s < 1.0, "{label}: out of sigmoid range: {s}");
        }
    }

    // ---------------------------------------------------------------- generic

    #[test]
    fn reports_the_accelerators_this_build_enables() {
        eprintln!(
            "hushar build enables: [{}]",
            onnxrt_rs::enabled_features().join(", ")
        );
        assert!(
            !onnxrt_rs::enabled_features().is_empty(),
            "a build with no accelerator could not serve anything"
        );
    }

    #[test]
    fn loads_a_model_and_reports_its_width() -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }

        let backend = load_onnx_model(&model, "cpu", 1, &[], None)?;
        assert_eq!(
            backend.inputs().len(),
            1,
            "the fixture declares a single input"
        );
        assert_eq!(
            backend.inputs()[0].width,
            Some(3),
            "the fixture takes 3 features"
        );
        assert_eq!(
            backend.inputs()[0].element_type,
            crate::inference::batch::ElementType::F32
        );
        assert!(
            backend.name().contains("onnxruntime"),
            "name should identify the engine, got {:?}",
            backend.name()
        );
        Ok(())
    }

    #[test]
    fn runs_a_multi_row_batch() -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }

        let backend = load_onnx_model(&model, "cpu", 1, &[], None)?;
        let batch = vec![1.0f32, 2.0, 3.0, 0.5, 1.0, 1.5];
        let predictions = run_rows(&backend, batch, 2)?;

        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].len(), 2, "the fixture emits 2 scores");
        for (row, scores) in predictions.iter().enumerate() {
            assert_sigmoid_range(&format!("row {row}"), scores);
        }
        Ok(())
    }

    #[test]
    fn a_malformed_device_id_is_reported() -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }
        let err = load_onnx_model(&model, "tensorrt:not-a-device", 1, &[], None)
            .expect_err("bad device id");
        assert!(
            err.to_string().contains("device id"),
            "unhelpful message: {err}"
        );
        Ok(())
    }

    /// The lower-level vendor providers must point at the supported spelling.
    #[test]
    fn cuda_and_rocm_point_at_the_supported_provider()
    -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }
        for (input, expected) in [("cuda", "tensorrt"), ("rocm", "migraphx")] {
            let err = load_onnx_model(&model, input, 1, &[], None).expect_err("should be rejected");
            assert!(
                err.to_string().contains(expected),
                "{input:?} should point at {expected:?}, got: {err}"
            );
        }
        Ok(())
    }

    /// A provider absent from this build must fail with a message naming what is
    /// available, rather than silently falling back to CPU.
    #[test]
    fn an_unavailable_provider_names_the_available_ones()
    -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }
        let err = load_onnx_model(&model, "NoSuchAccelerator", 1, &[], None)
            .expect_err("provider does not exist");
        let message = err.to_string();
        assert!(
            message.contains("not available") && message.contains("CPU"),
            "the error should list real providers: {message}"
        );
        Ok(())
    }

    #[test]
    fn invalid_model_bytes_are_reported() -> Result<(), crate::inference::InferenceError> {
        if !runtime_available() {
            return Ok(());
        }
        assert!(load_onnx_model(&[0, 1, 2, 3], "cpu", 1, &[], None).is_err());
        Ok(())
    }

    /// Many threads loading, running and dropping backends at once must work.
    ///
    /// The service shares one backend across request threads, and intends to
    /// support reloading a model while still serving from the old one, so
    /// concurrent load and teardown is a real scenario rather than a synthetic one.
    ///
    /// Note on scope: this does *not* reliably reproduce the ONNX Runtime
    /// "only one LoggingManager" failure that motivated `Environment::shared`.
    /// That needed a create to interleave with the teardown of the last
    /// outstanding environment, and it only surfaced when the whole suite ran in
    /// parallel -- running this test alone was verified not to trigger it even with
    /// the fix reverted. The invariant that actually prevents it is tested in
    /// `onnxrt-rs` (`shared_environment_is_created_once`); this test covers
    /// concurrent use.
    ///
    /// Each round drops its backend while another thread is loading, so a teardown
    /// races a load.
    #[test]
    fn models_can_be_loaded_and_dropped_concurrently()
    -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }

        let model = Arc::new(model);
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let model = Arc::clone(&model);
                std::thread::spawn(move || -> Result<(), String> {
                    for round in 0..12 {
                        let backend = load_onnx_model(&model, "cpu", 1, &[], None)
                            .map_err(|e| format!("thread {i} round {round} load: {e}"))?;
                        let predictions = run_rows(&backend, vec![1.0f32, 2.0, 3.0], 1)
                            .map_err(|e| format!("thread {i} round {round} run: {e}"))?;
                        if predictions[0].len() != 2 {
                            return Err(format!(
                                "thread {i} round {round}: expected width 2, got {}",
                                predictions[0].len()
                            ));
                        }
                    }
                    Ok(())
                })
            })
            .collect();

        for thread in threads {
            thread.join().expect("thread panicked")?;
        }
        Ok(())
    }

    // ------------------------------------------------------- per accelerator
    //
    // One module per accelerator, gated on the feature that enables it. Each asks
    // the same two questions of the backend: does it run, and does it agree with
    // CPU? Agreement is the one that earns its keep -- an accelerator that fails
    // to load is obvious, one that loads and quietly computes something else is not.

    #[cfg(feature = "cpu")]
    mod cpu {
        use super::*;

        #[test]
        fn cpu_runs_and_produces_sigmoid_output() {
            let Some(model) = test_model() else { return };
            if !runtime_available() {
                return;
            }
            let scores = scores_on(&model, "cpu").expect("CPU must always work");
            assert_sigmoid_range("cpu", &scores);
        }
    }

    #[cfg(feature = "coreml")]
    mod coreml {
        use super::*;

        #[test]
        fn coreml_agrees_with_cpu() {
            let Some(model) = test_model() else { return };
            if !runtime_available() {
                return;
            }
            let Some(scores) = scores_on(&model, "coreml") else {
                return;
            };
            assert_sigmoid_range("coreml", &scores);
            let baseline = scores_on(&model, "cpu").expect("CPU baseline");
            assert_matches_cpu("coreml", &scores, &baseline);
        }

        #[test]
        fn coreml_compute_units_agree_with_cpu() {
            let Some(model) = test_model() else { return };
            if !runtime_available() {
                return;
            }
            let baseline = scores_on(&model, "cpu").expect("CPU baseline");
            for spelling in [
                "coreml:all",
                "coreml:cpu_and_gpu",
                "coreml:cpu_and_neural_engine",
                "coreml:cpu_only",
            ] {
                let Some(scores) = scores_on(&model, spelling) else {
                    return;
                };
                assert_matches_cpu(spelling, &scores, &baseline);
            }
        }
    }

    #[cfg(feature = "xnnpack")]
    mod xnnpack {
        use super::*;

        #[test]
        fn xnnpack_agrees_with_cpu() {
            let Some(model) = test_model() else { return };
            if !runtime_available() {
                return;
            }
            let Some(scores) = scores_on(&model, "xnnpack") else {
                return;
            };
            assert_sigmoid_range("xnnpack", &scores);
            let baseline = scores_on(&model, "cpu").expect("CPU baseline");
            assert_matches_cpu("xnnpack", &scores, &baseline);
        }
    }

    #[cfg(feature = "tensorrt")]
    mod tensorrt {
        use super::*;

        /// TensorRT compiles the graph on first use, so this is slower than the
        /// others by design.
        #[test]
        fn tensorrt_agrees_with_cpu() {
            let Some(model) = test_model() else { return };
            if !runtime_available() {
                return;
            }
            let Some(scores) = scores_on(&model, "tensorrt") else {
                return;
            };
            assert_sigmoid_range("tensorrt", &scores);
            let baseline = scores_on(&model, "cpu").expect("CPU baseline");
            assert_matches_cpu("tensorrt", &scores, &baseline);
        }
    }

    #[cfg(feature = "migraphx")]
    mod migraphx {
        use super::*;

        /// MIGraphX is the one provider whose options struct this workspace fills
        /// by hand, ONNX Runtime exposing no factory for it, so a real run is the
        /// only way to know the defaults are right.
        #[test]
        fn migraphx_agrees_with_cpu() {
            let Some(model) = test_model() else { return };
            if !runtime_available() {
                return;
            }
            let Some(scores) = scores_on(&model, "migraphx") else {
                return;
            };
            assert_sigmoid_range("migraphx", &scores);
            let baseline = scores_on(&model, "cpu").expect("CPU baseline");
            assert_matches_cpu("migraphx", &scores, &baseline);
        }
    }
}
