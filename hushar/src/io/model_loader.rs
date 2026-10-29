// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Builds the inference backend from a model and the service configuration.

use std::sync::Arc;

use libnrt_rs::Runtime;
use onnxrt_rs::ExecutionProvider;

use crate::config::NeuronConfig;
use crate::inference::{InferenceBackend, NeuronBackend, OnnxRuntimeBackend};

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
/// * `session_options` - ONNX Runtime's string-keyed session settings, such as spinning
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
    parallelism: crate::config::Parallelism,
    session_options: &[(String, String)],
    mini_batch: Option<crate::inference::scoring::MiniBatch>,
) -> Result<Arc<dyn InferenceBackend>, crate::inference::InferenceError> {
    let provider: ExecutionProvider = execution_provider.parse()?;
    let backend = OnnxRuntimeBackend::load(
        model_bytes,
        &provider,
        parallelism,
        mini_batch,
        session_options,
    )?;
    Ok(Arc::new(backend))
}

/// Loads the same model into one session per entry in `session_options`.
///
/// The thread pools belong to the session, so a separate session per pool is what gives
/// concurrent requests pools that do not contend. The blocking pool is sized to this same
/// count, so each session has exactly one caller.
///
/// Loaded sequentially rather than together: the sessions are independent, but building
/// them at once would have each one's weight arena competing for the same memory bandwidth.
///
/// Every session reads the same bytes, so the weights are duplicated: memory is
/// `sessions x model size`. The graph and its signature are identical across the set.
///
/// # Errors
///
/// As [`load_onnx_model`]. The first failure is returned, naming the session that hit it.
pub fn load_onnx_sessions(
    model_bytes: &[u8],
    execution_provider: &str,
    parallelism: crate::config::Parallelism,
    session_options: &[Vec<(String, String)>],
    mini_batch: Option<crate::inference::scoring::MiniBatch>,
) -> Result<Vec<Arc<dyn InferenceBackend>>, crate::inference::InferenceError> {
    let total = session_options.len();
    let mut sessions = Vec::with_capacity(total);
    for (index, options) in session_options.iter().enumerate() {
        let backend = load_onnx_model(
            model_bytes,
            execution_provider,
            parallelism,
            options,
            mini_batch,
        )
        .map_err(|e| {
            crate::inference::InferenceError::from(format!("session {} of {total}: {e}", index + 1))
        })?;
        sessions.push(backend);
    }
    Ok(sessions)
}

/// The first NeuronCore of every replica.
///
/// Consecutive replicas share one core range until it holds `replicas_per_core` of them,
/// then the next range begins `cores_per_replica` cores later.
///
/// ```text
/// first 0, cores_per_replica 1, replicas_per_core 2, replicas 4  →  [0, 0, 1, 1]
/// first 0, cores_per_replica 2, replicas_per_core 1, replicas 2  →  [0, 2]
/// ```
///
/// # Errors
///
/// Fails when the arithmetic overflows, which no real device could satisfy.
pub(crate) fn replica_cores(
    first: usize,
    cores_per_replica: usize,
    replicas_per_core: usize,
    replicas: usize,
) -> Result<Vec<usize>, crate::inference::InferenceError> {
    (0..replicas)
        .map(|replica| {
            (replica / replicas_per_core.max(1))
                .checked_mul(cores_per_replica)
                .and_then(|offset| first.checked_add(offset))
                .ok_or_else(|| "the requested NeuronCore placement is too large".into())
        })
        .collect()
}

/// Loads a compiled NEFF into consecutive NeuronCore replicas.
///
/// `runtime` is shared across every Neuron model in the process because libnrt permits
/// exactly one initialized runtime. `replicas` is the service's
/// `threading.sessions_per_model`, so the same concurrency setting selects ORT sessions
/// or Neuron model replicas without applying ORT's thread pools to Neuron. Placement
/// follows [`replica_cores`].
///
/// # Errors
///
/// Fails when the runtime is unavailable, the requested core range is not visible, a
/// replica cannot load, or its tensors cannot be allocated.
pub fn load_neuron_replicas(
    neff: &[u8],
    config: &NeuronConfig,
    replicas: usize,
    mini_batch: Option<crate::inference::scoring::MiniBatch>,
    runtime: &mut Option<Arc<Runtime>>,
) -> Result<Vec<Arc<dyn InferenceBackend>>, crate::inference::InferenceError> {
    let runtime = match runtime {
        Some(runtime) => Arc::clone(runtime),
        slot @ None => {
            let initialized = Runtime::init()?;
            *slot = Some(Arc::clone(&initialized));
            initialized
        }
    };

    let replicas = replicas.max(1);
    let first = usize::try_from(config.start_neuron_core)
        .map_err(|_| "backend.start_neuron_core is not a valid index")?;
    let cores_per_replica = usize::try_from(config.neuron_cores_per_replica)
        .map_err(|_| "backend.neuron_cores_per_replica is not a valid count")?;
    let placement = replica_cores(
        first,
        cores_per_replica,
        config.replicas_per_neuron_core,
        replicas,
    )?;
    let last_core = placement
        .last()
        .copied()
        .unwrap_or(first)
        .checked_add(cores_per_replica)
        .ok_or("the requested NeuronCore range is too large")?;
    let visible = runtime.visible_neuron_cores()? as usize;
    if last_core > visible {
        return Err(format!(
            "{replicas} Neuron replica(s) starting at core {first} with {cores_per_replica} core(s) each and {} per core require cores {first}..{}, but libnrt reports {visible} visible NeuronCore(s)",
            config.replicas_per_neuron_core,
            last_core - 1
        )
        .into());
    }

    let mut loaded = Vec::with_capacity(replicas);
    for (replica, offset) in placement.into_iter().enumerate() {
        let start_neuron_core = i32::try_from(offset)
            .map_err(|_| "the requested NeuronCore index does not fit libnrt")?;
        let backend = NeuronBackend::load(
            Arc::clone(&runtime),
            neff,
            start_neuron_core,
            config.neuron_cores_per_replica,
            config.execution_contexts_per_replica,
            mini_batch,
        )
        .map_err(|error| -> crate::inference::InferenceError {
            format!(
                "Neuron replica {} of {replicas} on core {start_neuron_core}: {error}",
                replica + 1
            )
            .into()
        })?;
        loaded.push(Arc::new(backend) as Arc<dyn InferenceBackend>);
    }
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Reads the test fixture: 3 features in, 2 sigmoid outputs; `None` when absent.
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

    /// Whether an ONNX Runtime library is present; these tests skip when it is absent.
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
    /// Only the accelerator modules use this, so a build with no accelerator feature
    /// leaves it unused.
    #[allow(dead_code)]
    fn single_row() -> Vec<f32> {
        vec![1.0, 2.0, 3.0]
    }

    /// Runs `values` as `rows` rows through the backend's single input and returns
    /// the primary output, one `Vec` per row.
    ///
    /// The input's name, element type and shape come from the model, so this works for
    /// any single-input model.
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

    /// Scores for one row on `provider`, or `None` when the accelerator is absent from
    /// the loaded library or from this build.
    ///
    /// Only the accelerator modules use this, so a build with no accelerator feature
    /// leaves it unused.
    ///
    /// # Panics
    ///
    /// Panics when loading or inference fails for any other reason.
    #[allow(dead_code)]
    fn scores_on(model: &[u8], provider: &str) -> Option<Vec<f32>> {
        match load_onnx_model(
            model,
            provider,
            crate::config::Parallelism::Intra(1),
            &[],
            None,
        ) {
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

    /// Asserts an accelerator agrees with CPU, within a per-score tolerance of 1e-4.
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

    /// Consecutive replicas fill one core range before the next; one per range is the
    /// original consecutive placement.
    #[test]
    fn replicas_share_a_core_range_only_when_asked() {
        assert_eq!(replica_cores(0, 1, 1, 2).unwrap(), vec![0, 1]);
        assert_eq!(replica_cores(0, 1, 2, 4).unwrap(), vec![0, 0, 1, 1]);
        assert_eq!(replica_cores(0, 1, 2, 3).unwrap(), vec![0, 0, 1]);
        assert_eq!(replica_cores(1, 2, 1, 2).unwrap(), vec![1, 3]);
        assert_eq!(replica_cores(0, 2, 2, 4).unwrap(), vec![0, 0, 2, 2]);
        assert!(
            replica_cores(usize::MAX, 1, 1, 2).is_err(),
            "overflow is refused"
        );
    }

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

    /// Asserts one session is built per options entry, in order, for every session count
    /// the configuration can ask for.
    ///
    /// The length of `session_options` is how the session count reaches the loader, so this
    /// is what ties `sessions_per_model` to the number of engines that actually exist.
    #[test]
    fn builds_one_session_per_options_entry() -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }

        for count in [1usize, 2, 3] {
            let options = vec![Vec::new(); count];
            let sessions = load_onnx_sessions(
                &model,
                "cpu",
                crate::config::Parallelism::Intra(1),
                &options,
                None,
            )?;
            assert_eq!(
                sessions.len(),
                count,
                "{count} option set(s) should give {count} session(s)"
            );
            for session in &sessions {
                assert_eq!(
                    session.inputs().len(),
                    1,
                    "every replica loads the same graph"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn loads_a_model_and_reports_its_width() -> Result<(), crate::inference::InferenceError> {
        let Some(model) = test_model() else {
            return Ok(());
        };
        if !runtime_available() {
            return Ok(());
        }

        let backend = load_onnx_model(
            &model,
            "cpu",
            crate::config::Parallelism::Intra(1),
            &[],
            None,
        )?;
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

        let backend = load_onnx_model(
            &model,
            "cpu",
            crate::config::Parallelism::Intra(1),
            &[],
            None,
        )?;
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
        let err = load_onnx_model(
            &model,
            "tensorrt:not-a-device",
            crate::config::Parallelism::Intra(1),
            &[],
            None,
        )
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
            let err = load_onnx_model(
                &model,
                input,
                crate::config::Parallelism::Intra(1),
                &[],
                None,
            )
            .expect_err("should be rejected");
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
        let err = load_onnx_model(
            &model,
            "NoSuchAccelerator",
            crate::config::Parallelism::Intra(1),
            &[],
            None,
        )
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
        assert!(
            load_onnx_model(
                &[0, 1, 2, 3],
                "cpu",
                crate::config::Parallelism::Intra(1),
                &[],
                None
            )
            .is_err()
        );
        Ok(())
    }

    /// Many threads loading, running and dropping backends at once must work.
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
                        let backend = load_onnx_model(
                            &model,
                            "cpu",
                            crate::config::Parallelism::Intra(1),
                            &[],
                            None,
                        )
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

        /// TensorRT compiles the graph on first use, so this test is slower than the
        /// others.
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

        /// MIGraphX's options struct is filled in by hand, ONNX Runtime exposing no
        /// factory for it, so a real run is the only check that the defaults are right.
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
