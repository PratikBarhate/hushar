// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! One test module per accelerator, each gated on that accelerator's feature.
//!
//! ```bash
//! cargo test -p onnxrt-rs                            # CPU only (the default)
//! cargo test -p onnxrt-rs --features coreml          # CPU + CoreML
//! cargo test -p onnxrt-rs --features all-providers    # every accelerator
//! ```
//!
//! Each module asks the same four questions of its accelerator, so a new one has
//! an obvious shape to copy:
//!
//! 1. is the provider registered on a session without error?
//! 2. does a real inference run on it?
//! 3. does it agree with the CPU baseline?
//! 4. is its configuration surface (device id, compute units, threads) accepted?
//!
//! Question 3 is the one that matters most. A provider that fails to load is
//! obvious; a provider that loads and quietly computes something else is not.
//!
//! Enabling a feature is not a claim the hardware is present. A module whose
//! provider is missing from the loaded `libonnxruntime` skips with a reason —
//! see `common::with_provider`.
//!
//! NVIDIA is covered by TensorRT and AMD by MIGraphX, those being the vendors'
//! strategic inference stacks. The plain CUDA and ROCm providers are not offered;
//! `"cuda"` and `"rocm"` in configuration are rejected with a pointer to the
//! supported spelling rather than silently doing something else.

mod common;

// Which of these are used depends on which accelerator features are enabled, so
// a narrow build legitimately leaves some unused.
#[allow(unused_imports)]
use common::{
    api, assert_matches_cpu, assert_sigmoid_range, cpu_scores, scores_on, session_on, with_provider,
};
use onnxrt_rs::ExecutionProvider;

/// Names every accelerator this build enabled, so the log records what was
/// actually exercised rather than leaving it to be inferred from which tests ran.
#[test]
fn report_enabled_accelerators() {
    eprintln!(
        "this build enables: [{}]",
        onnxrt_rs::enabled_features().join(", ")
    );
    if let Some(api) = api() {
        eprintln!(
            "the loaded ONNX Runtime provides: [{}]",
            api.available_providers().unwrap_or_default().join(", ")
        );
    }
    assert!(
        onnxrt_rs::enabled_features().contains(&"cpu") || !cfg!(feature = "cpu"),
        "the cpu feature should report itself as enabled"
    );
}

/// Asking for an accelerator this build did not enable must fail with a message
/// naming the feature, distinctly from one the library lacks.
#[test]
fn a_disabled_accelerator_names_the_missing_feature() {
    let Some(_api) = api() else { return };

    // Pick a provider that this build did *not* enable. If everything is enabled
    // there is nothing to test, which is itself correct.
    let candidates = [
        ExecutionProvider::core_ml(),
        ExecutionProvider::xnnpack(),
        ExecutionProvider::tensor_rt(0),
        ExecutionProvider::migraphx(0),
    ];
    let Some(disabled) = candidates.into_iter().find(|p| !p.is_enabled()) else {
        eprintln!("skipping: this build enables every accelerator");
        return;
    };

    let feature = disabled.required_feature().expect("a gated provider");
    match session_on(&disabled) {
        Err(onnxrt_rs::Error::ProviderNotEnabled {
            provider,
            feature: reported,
            enabled,
        }) => {
            assert_eq!(provider, disabled.name());
            assert_eq!(reported, feature);
            assert!(
                !enabled.is_empty(),
                "the message should say what this build does enable"
            );
        }
        Err(e) => panic!("expected ProviderNotEnabled for {feature}, got: {e}"),
        Ok(_) => panic!("{feature} is disabled but a session was still built with it"),
    }
}

#[cfg(feature = "cpu")]
mod cpu {
    use super::*;

    /// CPU is the baseline every other accelerator is compared against, so it has
    /// to be right on its own terms first.
    #[test]
    fn cpu_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::Cpu, |provider| {
            let scores = scores_on(provider).expect("CPU inference");
            assert_sigmoid_range("CPU", &scores);
        });
    }

    /// CPU is present in every ONNX Runtime build, so unlike the accelerators this
    /// must not be skipped for lack of a provider.
    #[test]
    fn cpu_is_always_registerable() {
        let Some(_api) = api() else { return };
        session_on(&ExecutionProvider::Cpu).expect("CPU must always be usable");
    }
}

#[cfg(feature = "coreml")]
mod coreml {
    use super::*;
    use onnxrt_rs::{CoreMlComputeUnits, CoreMlModelFormat};

    #[test]
    fn coreml_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::core_ml(), |provider| {
            let scores = scores_on(provider).expect("CoreML inference");
            assert_sigmoid_range("CoreML", &scores);
        });
    }

    #[test]
    fn coreml_agrees_with_cpu() {
        with_provider(ExecutionProvider::core_ml(), |provider| {
            let actual = scores_on(provider).expect("CoreML inference");
            assert_matches_cpu("CoreML", &actual, &cpu_scores());
        });
    }

    /// Every compute-unit setting must be one CoreML accepts, and none may change
    /// the answer.
    ///
    /// The `MLComputeUnits` values are documented by ONNX Runtime but not by the
    /// vendored C header, so they are the one part of the provider mapping that
    /// could not be confirmed by reading the header. This pins them.
    #[test]
    fn every_coreml_compute_unit_setting_agrees_with_cpu() {
        with_provider(ExecutionProvider::core_ml(), |_| {
            let baseline = cpu_scores();
            for units in [
                CoreMlComputeUnits::All,
                CoreMlComputeUnits::CpuAndGpu,
                CoreMlComputeUnits::CpuAndNeuralEngine,
                CoreMlComputeUnits::CpuOnly,
            ] {
                let provider = ExecutionProvider::core_ml_with(units);
                let scores = scores_on(&provider)
                    .unwrap_or_else(|e| panic!("CoreML rejected {units:?}: {e}"));
                assert_sigmoid_range(&format!("CoreML {units:?}"), &scores);
                assert_matches_cpu(&format!("CoreML {units:?}"), &scores, &baseline);
            }
        });
    }

    /// Both model formats must run and agree with CPU.
    ///
    /// The format decides how much of a graph CoreML will take: `NeuralNetwork` has no
    /// N-dimensional `MatMul`, so it refuses a transformer's attention blocks and hands
    /// most of the graph back to CPU. That is a throughput question rather than a
    /// correctness one, which is exactly why it needs a test -- the wrong format is not
    /// visible in the answer.
    #[test]
    fn every_coreml_model_format_agrees_with_cpu() {
        with_provider(ExecutionProvider::core_ml(), |_| {
            let baseline = cpu_scores();
            for format in [
                CoreMlModelFormat::MlProgram,
                CoreMlModelFormat::NeuralNetwork,
            ] {
                let provider =
                    ExecutionProvider::core_ml_full(CoreMlComputeUnits::CpuAndNeuralEngine, format);
                let scores = scores_on(&provider)
                    .unwrap_or_else(|e| panic!("CoreML rejected {format:?}: {e}"));
                assert_sigmoid_range(&format!("CoreML {format:?}"), &scores);
                assert_matches_cpu(&format!("CoreML {format:?}"), &scores, &baseline);
            }
        });
    }
}

/// The configuration spellings, which need no OpenVINO present to check.
mod openvino_spellings {
    use onnxrt_rs::ExecutionProvider;

    fn parse(s: &str) -> (String, Option<u32>, Option<u32>) {
        match s
            .parse::<ExecutionProvider>()
            .expect("an OpenVINO provider")
        {
            ExecutionProvider::OpenVino {
                device_type,
                num_of_threads,
                num_streams,
            } => (device_type, num_of_threads, num_streams),
            other => panic!("{s:?} parsed as {other:?}"),
        }
    }

    /// CPU, because that is what the provider is reached for on a server. A GPU or NPU
    /// has to be named, so a configuration cannot land on one by accident.
    #[test]
    fn openvino_defaults_to_the_cpu_with_openvinos_own_thread_and_stream_defaults() {
        assert_eq!(parse("openvino"), ("CPU".to_owned(), None, None));
        assert_eq!(parse("open_vino"), ("CPU".to_owned(), None, None));
    }

    /// Uppercased, because OpenVINO's own device names are and a lowercase spelling in a
    /// configuration should not be a different thing.
    #[test]
    fn a_device_is_uppercased() {
        for (spelling, expected) in [
            ("openvino:cpu", "CPU"),
            ("openvino:gpu", "GPU"),
            ("openvino:npu", "NPU"),
            ("openvino:gpu.1", "GPU.1"),
            ("openvino:GPU", "GPU"),
        ] {
            assert_eq!(parse(spelling).0, expected, "{spelling}");
        }
    }

    #[test]
    fn threads_and_streams_are_optional_and_order_free() {
        assert_eq!(
            parse("openvino:cpu:threads=96"),
            ("CPU".to_owned(), Some(96), None)
        );
        assert_eq!(
            parse("openvino:streams=4:cpu"),
            ("CPU".to_owned(), None, Some(4))
        );
        assert_eq!(
            parse("openvino:threads=96:streams=4:gpu"),
            ("GPU".to_owned(), Some(96), Some(4))
        );
        // The upstream spellings work too, since that is what the docs name them.
        assert_eq!(
            parse("openvino:num_of_threads=8:num_streams=2"),
            ("CPU".to_owned(), Some(8), Some(2))
        );
    }

    /// A multi-device form carries a colon of its own, so it arrives as two tokens and
    /// has to be rejoined rather than read as two device types.
    #[test]
    fn a_multi_device_form_is_rejoined() {
        assert_eq!(parse("openvino:auto:gpu,cpu").0, "AUTO:GPU,CPU");
        assert_eq!(parse("openvino:hetero:gpu,cpu").0, "HETERO:GPU,CPU");
        assert_eq!(parse("openvino:multi:gpu,cpu:threads=8").0, "MULTI:GPU,CPU");
    }

    #[test]
    fn a_bad_spelling_is_refused_with_a_reason() {
        for (spelling, expected) in [
            ("openvino:cpu:precision=FP16", "unknown OpenVINO option"),
            ("openvino:cpu:threads=lots", "expected a positive number"),
            ("openvino:cpu:threads=0", "must be positive"),
            ("openvino:cpu:gpu", "two device types"),
        ] {
            let err = spelling
                .parse::<ExecutionProvider>()
                .expect_err("should be refused")
                .to_string();
            assert!(
                err.contains(expected),
                "{spelling} should mention {expected:?}: {err}"
            );
        }
    }

    /// The banner prints this, so it has to say what was configured.
    #[test]
    fn the_description_names_the_device_and_any_options() {
        let described = |s: &str| s.parse::<ExecutionProvider>().unwrap().to_string();
        assert_eq!(described("openvino"), "OpenVINO(CPU)");
        assert_eq!(
            described("openvino:cpu:threads=96:streams=2"),
            "OpenVINO(CPU, threads=96, streams=2)"
        );
    }
}

/// The configuration spellings, which need no CoreML present to check.
mod coreml_spellings {
    use onnxrt_rs::{CoreMlComputeUnits, CoreMlModelFormat, ExecutionProvider};

    fn parse(s: &str) -> (CoreMlComputeUnits, CoreMlModelFormat) {
        match s.parse::<ExecutionProvider>().expect("a CoreML provider") {
            ExecutionProvider::CoreMl {
                compute_units,
                model_format,
            } => (compute_units, model_format),
            other => panic!("{s:?} parsed as {other:?}"),
        }
    }

    /// ML Program is the default, because the older format cannot take a transformer.
    #[test]
    fn coreml_defaults_to_the_ml_program_format() {
        assert_eq!(
            parse("coreml"),
            (CoreMlComputeUnits::All, CoreMlModelFormat::MlProgram)
        );
        assert_eq!(
            parse("coreml:cpu_and_neural_engine"),
            (
                CoreMlComputeUnits::CpuAndNeuralEngine,
                CoreMlModelFormat::MlProgram
            )
        );
    }

    /// Compute units and format are independent, so neither has to be named to reach
    /// the other and the order between them does not matter.
    #[test]
    fn the_two_coreml_options_are_independent_and_unordered() {
        let restricted = (
            CoreMlComputeUnits::CpuAndNeuralEngine,
            CoreMlModelFormat::NeuralNetwork,
        );
        assert_eq!(
            parse("coreml:cpu_and_neural_engine:neuralnetwork"),
            restricted
        );
        assert_eq!(
            parse("coreml:neuralnetwork:cpu_and_neural_engine"),
            restricted
        );
        assert_eq!(
            parse("coreml:neuralnetwork"),
            (CoreMlComputeUnits::All, CoreMlModelFormat::NeuralNetwork)
        );
    }

    /// A misspelling names the value and lists what was expected, because a provider
    /// that silently ignored it would run slowly rather than fail.
    #[test]
    fn an_unknown_coreml_option_is_refused_with_the_alternatives() {
        let error = "coreml:cpu_and_ane"
            .parse::<ExecutionProvider>()
            .expect_err("an unknown option must be refused");
        let message = error.to_string();
        assert!(
            message.contains("cpu_and_ane"),
            "names the input: {message}"
        );
        assert!(
            message.contains("mlprogram"),
            "lists the formats: {message}"
        );
        assert!(
            message.contains("cpu_and_neural_engine"),
            "lists the compute units: {message}"
        );
    }
}

#[cfg(feature = "xnnpack")]
mod xnnpack {
    use super::*;

    #[test]
    fn xnnpack_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::xnnpack(), |provider| {
            let scores = scores_on(provider).expect("XNNPACK inference");
            assert_sigmoid_range("XNNPACK", &scores);
        });
    }

    #[test]
    fn xnnpack_agrees_with_cpu() {
        with_provider(ExecutionProvider::xnnpack(), |provider| {
            let actual = scores_on(provider).expect("XNNPACK inference");
            assert_matches_cpu("XNNPACK", &actual, &cpu_scores());
        });
    }

    /// XNNPACK takes its own thread-pool size.
    #[test]
    fn xnnpack_accepts_an_explicit_thread_count() {
        with_provider(ExecutionProvider::xnnpack(), |_| {
            let provider = ExecutionProvider::Xnnpack {
                intra_op_threads: Some(2),
            };
            let scores = scores_on(&provider).expect("XNNPACK with 2 threads");
            assert_matches_cpu("XNNPACK(threads=2)", &scores, &cpu_scores());
        });
    }
}

#[cfg(feature = "tensorrt")]
mod tensorrt {
    use super::*;

    /// TensorRT compiles the graph on first use, so this is slower than the others
    /// by design.
    #[test]
    fn tensorrt_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::tensor_rt(0), |provider| {
            let scores = scores_on(provider).expect("TensorRT inference");
            assert_sigmoid_range("TensorRT", &scores);
        });
    }

    #[test]
    fn tensorrt_agrees_with_cpu() {
        with_provider(ExecutionProvider::tensor_rt(0), |provider| {
            let actual = scores_on(provider).expect("TensorRT inference");
            assert_matches_cpu("TensorRT", &actual, &cpu_scores());
        });
    }
}

#[cfg(feature = "migraphx")]
mod migraphx {
    use super::*;

    /// MIGraphX is the one provider whose options struct this crate fills by
    /// hand, because ONNX Runtime exposes no factory for it. That makes a real run
    /// the only way to know the defaults are right — in particular
    /// `migraphx_mem_limit`, where zero would mean zero bytes rather than
    /// unlimited.
    #[test]
    fn migraphx_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::migraphx(0), |provider| {
            let scores = scores_on(provider).expect("MIGraphX inference");
            assert_sigmoid_range("MIGraphX", &scores);
        });
    }

    #[test]
    fn migraphx_agrees_with_cpu() {
        with_provider(ExecutionProvider::migraphx(0), |provider| {
            let actual = scores_on(provider).expect("MIGraphX inference");
            assert_matches_cpu("MIGraphX", &actual, &cpu_scores());
        });
    }
}
