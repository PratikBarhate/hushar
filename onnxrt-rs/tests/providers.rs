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
//! Each module asks the same four questions of its accelerator:
//!
//! 1. is the provider registered on a session without error?
//! 2. does a real inference run on it?
//! 3. does it agree with the CPU baseline?
//! 4. is its configuration surface (device id, compute units, threads) accepted?
//!
//! Enabling a feature is not a claim the hardware is present. A module whose
//! provider is missing from the loaded `libonnxruntime` skips with a reason —
//! see `common::with_provider`.
//!
//! The plain CUDA and ROCm providers are not offered: NVIDIA is covered by TensorRT
//! and AMD by MIGraphX, and `"cuda"` or `"rocm"` in configuration is rejected with a
//! pointer to the supported spelling.

mod common;

// A narrow build legitimately leaves some of these unused.
#[allow(unused_imports)]
use common::{
    api, assert_matches_cpu, assert_sigmoid_range, cpu_scores, scores_on, session_on, with_provider,
};
use onnxrt_rs::ExecutionProvider;

/// Logs the accelerators this build enables, and asserts the `cpu` feature reports
/// itself as enabled.
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

/// Asserts each provider spelling reports whether it runs on a device of its own.
///
/// Needs no library: this is a property of the parsed provider, and callers use it to
/// decide what element types can reach the model. CoreML and OpenVINO are either one, so
/// the units they were pointed at decide, and a custom name counts as a host provider
/// because nothing about it says otherwise.
#[test]
fn providers_report_whether_they_run_off_host() {
    for spelling in [
        "cpu",
        "xnnpack",
        "xnnpack:4",
        "coreml:cpu_only",
        "openvino:cpu",
        "openvino:CPU",
        "QNNExecutionProvider",
    ] {
        let provider: ExecutionProvider = spelling.parse().expect("a known spelling");
        assert!(
            !provider.runs_off_host(),
            "{spelling} runs ONNX Runtime's own kernels on the host"
        );
    }

    for spelling in [
        "tensorrt",
        "trt:1",
        "migraphx",
        "migraphx:0",
        "coreml",
        "coreml:cpu_and_gpu",
        "openvino:gpu",
        "openvino:npu",
    ] {
        let provider: ExecutionProvider = spelling.parse().expect("a known spelling");
        assert!(
            provider.runs_off_host(),
            "{spelling} executes on a device with memory of its own"
        );
    }
}

/// Asserts a provider this build did not enable fails with `ProviderNotEnabled`
/// naming the missing feature, distinctly from one the library lacks.///
/// Skips when this build enables every accelerator.
#[test]
fn a_disabled_accelerator_names_the_missing_feature() {
    let Some(_api) = api() else { return };

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

    /// Asserts CPU inference runs and produces sigmoid outputs; it is the baseline
    /// every other accelerator is compared against.
    #[test]
    fn cpu_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::Cpu, |provider| {
            let scores = scores_on(provider).expect("CPU inference");
            assert_sigmoid_range("CPU", &scores);
        });
    }

    /// Asserts CPU registers on any ONNX Runtime build, so this never skips for a
    /// missing provider.
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

    /// Asserts CoreML inference runs and produces sigmoid outputs.
    #[test]
    fn coreml_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::core_ml(), |provider| {
            let scores = scores_on(provider).expect("CoreML inference");
            assert_sigmoid_range("CoreML", &scores);
        });
    }

    /// Asserts CoreML scores agree with the CPU baseline.
    #[test]
    fn coreml_agrees_with_cpu() {
        with_provider(ExecutionProvider::core_ml(), |provider| {
            let actual = scores_on(provider).expect("CoreML inference");
            assert_matches_cpu("CoreML", &actual, &cpu_scores());
        });
    }

    /// Asserts every compute-unit setting is accepted by CoreML and none changes the
    /// answer.
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

    /// Asserts both CoreML model formats run and agree with CPU.
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

    /// Asserts a bare `openvino` spelling defaults to CPU with OpenVINO's own thread
    /// and stream defaults; a GPU or NPU has to be named.
    #[test]
    fn openvino_defaults_to_the_cpu_with_openvinos_own_thread_and_stream_defaults() {
        assert_eq!(parse("openvino"), ("CPU".to_owned(), None, None));
        assert_eq!(parse("open_vino"), ("CPU".to_owned(), None, None));
    }

    /// Asserts a device name is uppercased, matching OpenVINO's own spellings.
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

    /// Asserts threads and streams are optional and order-free, in this crate's and
    /// OpenVINO's own spellings.
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
        assert_eq!(
            parse("openvino:num_of_threads=8:num_streams=2"),
            ("CPU".to_owned(), Some(8), Some(2))
        );
    }

    /// Asserts a multi-device form is rejoined rather than read as two device types,
    /// its own colon notwithstanding.
    #[test]
    fn a_multi_device_form_is_rejoined() {
        assert_eq!(parse("openvino:auto:gpu,cpu").0, "AUTO:GPU,CPU");
        assert_eq!(parse("openvino:hetero:gpu,cpu").0, "HETERO:GPU,CPU");
        assert_eq!(parse("openvino:multi:gpu,cpu:threads=8").0, "MULTI:GPU,CPU");
    }

    /// Asserts a bad spelling is refused with an error naming the reason.
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

    /// Asserts the description, which the banner prints, names the device and any
    /// options.
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

/// The configuration spellings, which need no TensorRT present to check.
mod tensorrt_spellings {
    use onnxrt_rs::ExecutionProvider;

    fn parse(s: &str) -> (i32, Vec<(String, String)>) {
        match s.parse::<ExecutionProvider>().expect("a TensorRT provider") {
            ExecutionProvider::TensorRt { device_id, options } => (device_id, options),
            other => panic!("{s:?} parsed as {other:?}"),
        }
    }

    fn pairs(kv: &[(&str, &str)]) -> Vec<(String, String)> {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Asserts a bare spelling is device 0 with ONNX Runtime's own defaults, and a
    /// device id is still read as before.
    #[test]
    fn tensorrt_defaults_to_device_zero_with_no_options() {
        assert_eq!(parse("tensorrt"), (0, Vec::new()));
        assert_eq!(parse("trt"), (0, Vec::new()));
        assert_eq!(parse("trt:1"), (1, Vec::new()));
        assert_eq!(parse("TensorRT:0"), (0, Vec::new()));
    }

    /// Asserts the short spellings expand to the keys ONNX Runtime reads, in any
    /// order and with or without a device id.
    #[test]
    fn the_short_spellings_expand_to_onnx_runtime_keys() {
        assert_eq!(
            parse("tensorrt:0:cache=/var/cache/trt"),
            (
                0,
                pairs(&[
                    ("trt_engine_cache_enable", "1"),
                    ("trt_engine_cache_path", "/var/cache/trt"),
                ])
            )
        );
        assert_eq!(
            parse("trt:fp16=1:1"),
            (1, pairs(&[("trt_fp16_enable", "1")]))
        );
        assert_eq!(
            parse("trt:fp16=off:workspace=1073741824:opt_level=3"),
            (
                0,
                pairs(&[
                    ("trt_fp16_enable", "0"),
                    ("trt_max_workspace_size", "1073741824"),
                    ("trt_builder_optimization_level", "3"),
                ])
            )
        );
        assert_eq!(
            parse("trt:timing_cache=/tmp/timing"),
            (
                0,
                pairs(&[
                    ("trt_timing_cache_enable", "1"),
                    ("trt_timing_cache_path", "/tmp/timing"),
                ])
            )
        );
    }

    /// Asserts a `trt_*` key is passed through verbatim, so an option this crate does
    /// not spell short is still reachable.
    #[test]
    fn a_trt_key_is_passed_through_verbatim() {
        assert_eq!(
            parse("tensorrt:trt_cuda_graph_enable=1:trt_dump_subgraphs=0"),
            (
                0,
                pairs(&[("trt_cuda_graph_enable", "1"), ("trt_dump_subgraphs", "0"),])
            )
        );
    }

    /// Asserts a bad spelling is refused with an error naming the reason.
    #[test]
    fn a_bad_spelling_is_refused_with_a_reason() {
        for (spelling, expected) in [
            ("tensorrt:gpu0", "expected a device id"),
            ("tensorrt:0:1", "two device ids"),
            ("tensorrt:precision=fp16", "unknown TensorRT option"),
            ("tensorrt:fp16=maybe", "expected 0 or 1"),
            ("tensorrt:cache=", "has no value"),
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

    /// Asserts the description, which the banner prints, names the device and every
    /// option as ONNX Runtime received it.
    #[test]
    fn the_description_names_the_device_and_every_option() {
        let described = |s: &str| s.parse::<ExecutionProvider>().unwrap().to_string();
        assert_eq!(described("trt"), "TensorRT(device=0)");
        assert_eq!(
            described("tensorrt:1:cache=/c:fp16=1"),
            "TensorRT(device=1, trt_engine_cache_enable=1, trt_engine_cache_path=/c, \
             trt_fp16_enable=1)"
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

    /// Asserts CoreML defaults to the ML Program format, whatever the compute units.
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

    /// Asserts compute units and model format are independent and may be given in
    /// either order.
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

    /// Asserts an unknown CoreML option is refused with the input named and the
    /// accepted values listed.
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

    /// Asserts XNNPACK inference runs and produces sigmoid outputs.
    #[test]
    fn xnnpack_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::xnnpack(), |provider| {
            let scores = scores_on(provider).expect("XNNPACK inference");
            assert_sigmoid_range("XNNPACK", &scores);
        });
    }

    /// Asserts XNNPACK scores agree with the CPU baseline.
    #[test]
    fn xnnpack_agrees_with_cpu() {
        with_provider(ExecutionProvider::xnnpack(), |provider| {
            let actual = scores_on(provider).expect("XNNPACK inference");
            assert_matches_cpu("XNNPACK", &actual, &cpu_scores());
        });
    }

    /// Asserts XNNPACK accepts its own thread-pool size and still agrees with CPU.
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

    /// Asserts TensorRT inference runs and produces sigmoid outputs; slower than the
    /// other providers, since TensorRT compiles the graph on first use.
    #[test]
    fn tensorrt_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::tensor_rt(0), |provider| {
            let scores = scores_on(provider).expect("TensorRT inference");
            assert_sigmoid_range("TensorRT", &scores);
        });
    }

    /// Asserts TensorRT scores agree with the CPU baseline.
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

    /// Asserts MIGraphX inference runs and produces sigmoid outputs, which is the only
    /// check on this crate's hand-filled options struct — ONNX Runtime exposes no
    /// factory for it, and a zero `migraphx_mem_limit` would mean zero bytes rather
    /// than unlimited.
    #[test]
    fn migraphx_runs_and_produces_sigmoid_output() {
        with_provider(ExecutionProvider::migraphx(0), |provider| {
            let scores = scores_on(provider).expect("MIGraphX inference");
            assert_sigmoid_range("MIGraphX", &scores);
        });
    }

    /// Asserts MIGraphX scores agree with the CPU baseline.
    #[test]
    fn migraphx_agrees_with_cpu() {
        with_provider(ExecutionProvider::migraphx(0), |provider| {
            let actual = scores_on(provider).expect("MIGraphX inference");
            assert_matches_cpu("MIGraphX", &actual, &cpu_scores());
        });
    }
}
