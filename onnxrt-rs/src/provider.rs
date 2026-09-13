// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Execution provider selection.
//!
//! An *execution provider* (EP) is ONNX Runtime's term for a hardware backend.
//! Registering one asks the runtime to place as much of the graph as it can onto
//! that device, leaving whatever it cannot handle on the CPU. Registering none
//! runs everything on the CPU.
//!
//! # Availability is a property of the loaded library, not of this crate
//!
//! This is the thing to internalise before using this module. An EP is compiled
//! *into* `libonnxruntime`, so which ones you can use is decided by the build you
//! ship, not by any Cargo feature here:
//!
//! | Distribution | Providers compiled in |
//! |---|---|
//! | `onnxruntime-osx-*` | CPU, CoreML |
//! | `onnxruntime-linux-x64`, `onnxruntime-linux-aarch64` | CPU |
//! | `onnxruntime-win-x64` | CPU |
//! | `onnxruntime-{linux,win}-x64-gpu_cuda12`, `-gpu_cuda13` | CPU, CUDA, TensorRT |
//! | ROCm / MIGraphX builds | CPU, ROCm, MIGraphX — from source or AMD's packages |
//! | XNNPACK | any build configured with `--use_xnnpack` |
//!
//! Because the library is resolved at run time, the honest answer to "is TensorRT
//! available?" can only be obtained at run time. Ask
//! [`Api::available_providers`]. [`ExecutionProvider::append`] checks against that
//! list first, so requesting a provider the loaded library lacks produces
//! [`Error::ProviderUnavailable`], naming what *is* available, instead of an
//! opaque ONNX Runtime status.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::str::FromStr;

use crate::api::{Api, ort_fn};
use crate::error::{Error, Result};
use crate::sys;

/// Creates a provider-options object through ONNX Runtime's own factory, applies
/// string options to it, registers it, and releases it on every path.
///
/// Using the factory rather than filling a struct means ONNX Runtime supplies the
/// defaults for every field this crate does not set — which matters because none
/// of the providers that use this path can be tested on a developer machine.
macro_rules! append_via_factory {
    ($api:expr, $options:expr, $kv:expr, $create:ident, $update:ident, $release:ident, $append:ident) => {{
        let api: Api = $api;
        let create = ort_fn!(api, $create);
        let update = ort_fn!(api, $update);
        let append = ort_fn!(api, $append);

        let mut provider_options = ptr::null_mut();
        // SAFETY: valid out-parameter; on success ONNX Runtime writes an owned
        // options object, released below.
        unsafe { api.check(create(&mut provider_options))? };

        // Everything from here must run to the release, so failures are captured
        // rather than propagated with `?`.
        let outcome = (|| -> Result<()> {
            let (keys, values) = c_pairs($kv)?;
            if !keys.is_empty() {
                let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
                let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
                // SAFETY: options object is live; keys and values outlive the call.
                unsafe {
                    api.check(update(
                        provider_options,
                        key_ptrs.as_ptr(),
                        value_ptrs.as_ptr(),
                        key_ptrs.len(),
                    ))?
                };
            }
            // SAFETY: both pointers are live. ONNX Runtime copies what it needs,
            // so releasing afterwards is correct.
            unsafe { api.check(append($options, provider_options)) }
        })();

        if let Some(release) = api.raw().$release {
            // SAFETY: created by the matching factory, released exactly once.
            unsafe { release(provider_options) };
        }
        outcome
    }};
}

/// Which compute units the CoreML provider may use.
///
/// Maps to CoreML's `MLComputeUnits` provider option. `CpuAndNeuralEngine` is
/// often the right choice on Apple Silicon for small models: the Neural Engine
/// has far lower latency than the GPU once the model is resident, and avoids
/// competing with the display for GPU time.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CoreMlComputeUnits {
    /// CPU, GPU and Neural Engine. CoreML decides placement.
    #[default]
    All,
    /// CPU and GPU only.
    CpuAndGpu,
    /// CPU and Neural Engine only.
    CpuAndNeuralEngine,
    /// CPU only. Useful for isolating whether an accuracy change comes from the
    /// accelerator.
    CpuOnly,
}

impl CoreMlComputeUnits {
    /// The string CoreML expects for `MLComputeUnits`.
    fn option_value(self) -> &'static str {
        match self {
            Self::All => "ALL",
            Self::CpuAndGpu => "CPUAndGPU",
            Self::CpuAndNeuralEngine => "CPUAndNeuralEngine",
            Self::CpuOnly => "CPUOnly",
        }
    }
}

/// Which CoreML model format the provider converts a graph into.
///
/// `MlProgram` is the default because `NeuralNetwork` cannot express an N-dimensional
/// `MatMul`, so it refuses the attention blocks of a transformer and hands most of the
/// graph back.
///
/// `NeuralNetwork` is the older format and remains ONNX Runtime's own default; it is
/// kept here for a host too old for `MlProgram`, which needs macOS 12 or iOS 15.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CoreMlModelFormat {
    /// CoreML's ML Program format. Supports N-dimensional `MatMul`.
    #[default]
    MlProgram,
    /// CoreML's older NeuralNetwork format.
    NeuralNetwork,
}

impl CoreMlModelFormat {
    /// The string CoreML expects for `ModelFormat`.
    fn option_value(self) -> &'static str {
        match self {
            Self::MlProgram => "MLProgram",
            Self::NeuralNetwork => "NeuralNetwork",
        }
    }
}

/// A hardware backend to register on a session.
///
/// Construct these with the helpers ([`ExecutionProvider::core_ml`],
/// [`ExecutionProvider::tensor_rt`], …) or parse one from configuration with
/// [`str::parse`]; see the [`FromStr`] implementation for the accepted spellings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionProvider {
    /// The default CPU provider. Always available, and always the final fallback
    /// even when another provider is registered, so there is rarely a reason to
    /// name it explicitly.
    Cpu,

    /// Apple CoreML — Apple Silicon GPU and Neural Engine.
    ///
    /// Fields:
    /// - `compute_units` — which compute units CoreML may use.
    /// - `model_format` — which CoreML format the graph is converted into.
    CoreMl {
        compute_units: CoreMlComputeUnits,
        model_format: CoreMlModelFormat,
    },

    /// XNNPACK — heavily optimised CPU kernels, most valuable on ARM.
    ///
    /// Not in the default ONNX Runtime builds; needs one configured with
    /// `--use_xnnpack`.
    Xnnpack {
        /// XNNPACK's own thread-pool size. `None` uses the session's.
        intra_op_threads: Option<u32>,
    },

    /// NVIDIA, through TensorRT.
    ///
    /// TensorRT compiles the graph ahead of first use, so the first inference is
    /// slow and later ones are usually the fastest available on NVIDIA hardware.
    /// It is NVIDIA's strategic inference stack, which is why it is the NVIDIA
    /// option here rather than the plain CUDA provider.
    ///
    /// Note the coverage tradeoff: TensorRT does not implement every ONNX
    /// operator, and nodes it cannot take fall back to **CPU**, which means
    /// host-device copies mid-graph. For a fully supported graph that costs
    /// nothing; for a partially supported one it can be slower than CUDA would
    /// have been.
    TensorRt {
        /// CUDA device ordinal.
        device_id: i32,
    },

    /// AMD, through MIGraphX.
    ///
    /// MIGraphX is AMD's graph-optimising inference compiler, layered over ROCm,
    /// and is the AMD option here for the same reason TensorRT is the NVIDIA one.
    /// The same fallback-to-CPU caveat as [`Self::TensorRt`] applies.
    MiGraphx {
        /// HIP device ordinal.
        device_id: i32,
    },

    /// Intel's OpenVINO, on Intel CPU, integrated or discrete GPU, or NPU.
    ///
    /// The provider Intel tunes for its own hardware, and the one to reach for on a Xeon
    /// rather than the default CPU provider. Not in any prebuilt ONNX Runtime archive: it
    /// needs a source build with `--use_openvino`, and OpenVINO itself installed.
    OpenVino {
        /// OpenVINO's `device_type`. `CPU`, `GPU`, `NPU`, `GPU.0`, or one of the
        /// multi-device forms such as `AUTO:GPU,CPU`.
        device_type: String,
        /// `num_of_threads`. Inference threads OpenVINO may use, `None` for its own
        /// default of 8.
        ///
        /// **Deprecated upstream** since ONNX Runtime 1.23 in favour of `load_config`
        /// with `INFERENCE_NUM_THREADS`. Still accepted, and the only way to reach the
        /// setting from a provider string.
        num_of_threads: Option<u32>,
        /// `num_streams`. Parallel inference streams, `None` for its own default of 1,
        /// which is the latency-oriented choice.
        ///
        /// Deprecated upstream alongside `num_of_threads`, in favour of `NUM_STREAMS`.
        num_streams: Option<u32>,
    },

    /// Any provider by name, with raw string options.
    ///
    /// An escape hatch for providers this enum does not model (QNN, OpenVINO,
    /// WebGPU, VitisAI) and for options added after this crate was written.
    Custom {
        /// Either the short or the full registry name, for example `"QNN"` or
        /// `"QNNExecutionProvider"`.
        name: String,
        /// Provider options, passed through untouched.
        options: Vec<(String, String)>,
    },
}

impl ExecutionProvider {
    /// OpenVINO on the CPU, with OpenVINO's own thread and stream defaults.
    pub fn open_vino() -> Self {
        Self::OpenVino {
            device_type: "CPU".to_owned(),
            num_of_threads: None,
            num_streams: None,
        }
    }

    /// CoreML with default compute units and format.
    pub fn core_ml() -> Self {
        Self::CoreMl {
            compute_units: CoreMlComputeUnits::default(),
            model_format: CoreMlModelFormat::default(),
        }
    }

    /// CoreML restricted to specific compute units, in the default format.
    pub fn core_ml_with(compute_units: CoreMlComputeUnits) -> Self {
        Self::CoreMl {
            compute_units,
            model_format: CoreMlModelFormat::default(),
        }
    }

    /// CoreML with both the compute units and the format chosen.
    pub fn core_ml_full(
        compute_units: CoreMlComputeUnits,
        model_format: CoreMlModelFormat,
    ) -> Self {
        Self::CoreMl {
            compute_units,
            model_format,
        }
    }

    /// XNNPACK using the session's thread-pool size.
    pub fn xnnpack() -> Self {
        Self::Xnnpack {
            intra_op_threads: None,
        }
    }

    /// TensorRT on `device_id`.
    pub fn tensor_rt(device_id: i32) -> Self {
        Self::TensorRt { device_id }
    }

    /// MIGraphX on `device_id`.
    pub fn migraphx(device_id: i32) -> Self {
        Self::MiGraphx { device_id }
    }

    /// Short human-readable name, for logs and metric dimensions.
    pub fn name(&self) -> &str {
        match self {
            Self::Cpu => "CPU",
            Self::CoreMl { .. } => "CoreML",
            Self::Xnnpack { .. } => "XNNPACK",
            Self::OpenVino { .. } => "OpenVINO",
            Self::TensorRt { .. } => "TensorRT",
            Self::MiGraphx { .. } => "MIGraphX",
            Self::Custom { name, .. } => name,
        }
    }

    /// The name ONNX Runtime reports from [`Api::available_providers`].
    ///
    /// Always the long `…ExecutionProvider` form, which is what the registry
    /// uses, even though the append call accepts the short form too.
    pub fn registry_name(&self) -> &str {
        match self {
            Self::Cpu => "CPUExecutionProvider",
            Self::CoreMl { .. } => "CoreMLExecutionProvider",
            Self::Xnnpack { .. } => "XnnpackExecutionProvider",
            Self::OpenVino { .. } => "OpenVINOExecutionProvider",
            Self::TensorRt { .. } => "TensorrtExecutionProvider",
            Self::MiGraphx { .. } => "MIGraphXExecutionProvider",
            Self::Custom { name, .. } => name,
        }
    }

    /// Whether the loaded ONNX Runtime was built with this provider.
    ///
    /// Compares with the `ExecutionProvider` suffix stripped and case folded, so
    /// a [`Self::Custom`] named either `"QNN"` or `"QNNExecutionProvider"` is
    /// recognised.
    pub fn is_available(&self, api: Api) -> Result<bool> {
        let available = api.available_providers()?;
        let wanted = normalized_provider_name(self.registry_name());
        Ok(available
            .iter()
            .any(|p| normalized_provider_name(p) == wanted))
    }

    /// The Cargo feature that enables this provider, if it is gated by one.
    ///
    /// [`Self::Custom`] is not gated: it is the escape hatch for providers this
    /// enum does not model, so a feature list could not usefully cover it.
    pub fn required_feature(&self) -> Option<&'static str> {
        match self {
            Self::Cpu => Some("cpu"),
            Self::CoreMl { .. } => Some("coreml"),
            Self::Xnnpack { .. } => Some("xnnpack"),
            Self::OpenVino { .. } => Some("openvino"),
            Self::TensorRt { .. } => Some("tensorrt"),
            Self::MiGraphx { .. } => Some("migraphx"),
            Self::Custom { .. } => None,
        }
    }

    /// Whether this build was compiled with this provider enabled.
    ///
    /// This is a compile-time property of *this crate*, and is separate from
    /// [`Self::is_available`], which asks the loaded `libonnxruntime` what it was
    /// built with. Both must hold: the feature says the deployment intends to use
    /// the hardware, the runtime check says the library can.
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::Cpu => cfg!(feature = "cpu"),
            Self::CoreMl { .. } => cfg!(feature = "coreml"),
            Self::Xnnpack { .. } => cfg!(feature = "xnnpack"),
            Self::OpenVino { .. } => cfg!(feature = "openvino"),
            Self::TensorRt { .. } => cfg!(feature = "tensorrt"),
            Self::MiGraphx { .. } => cfg!(feature = "migraphx"),
            Self::Custom { .. } => true,
        }
    }

    /// Registers this provider on `options`.
    ///
    /// Two checks happen first, in order, because they fail for different reasons
    /// and the messages need to say which:
    ///
    /// 1. [`Self::is_enabled`] — was this crate compiled with the provider's
    ///    feature? If not, [`Error::ProviderNotEnabled`].
    /// 2. [`Self::is_available`] — was the loaded `libonnxruntime` built with it?
    ///    If not, [`Error::ProviderUnavailable`], naming what is available.
    pub(crate) fn append(&self, api: Api, options: *mut sys::OrtSessionOptions) -> Result<()> {
        if !self.is_enabled() {
            return Err(Error::ProviderNotEnabled {
                provider: self.name().to_owned(),
                feature: self.required_feature().unwrap_or("<none>"),
                enabled: enabled_features().join(", "),
            });
        }

        // The CPU provider is implicit and is not always listed in the registry,
        // so short-circuit rather than checking availability.
        if matches!(self, Self::Cpu) {
            return Ok(());
        }

        if !self.is_available(api)? {
            return Err(Error::ProviderUnavailable {
                provider: self.name().to_owned(),
                available: api.available_providers()?.join(", "),
            });
        }

        match self {
            Self::Cpu => Ok(()),

            // CoreML and XNNPACK have no dedicated appender; the header directs
            // them through the generic string-keyed one.
            Self::CoreMl {
                compute_units,
                model_format,
            } => append_by_name(
                api,
                options,
                "CoreML",
                &[
                    (
                        "MLComputeUnits".to_owned(),
                        compute_units.option_value().to_owned(),
                    ),
                    (
                        "ModelFormat".to_owned(),
                        model_format.option_value().to_owned(),
                    ),
                ],
            ),
            Self::Xnnpack { intra_op_threads } => {
                let opts = match intra_op_threads {
                    Some(n) => vec![("intra_op_num_threads".to_owned(), n.to_string())],
                    None => Vec::new(),
                };
                append_by_name(api, options, "XNNPACK", &opts)
            }
            // Through the generic appender, like CoreML and XNNPACK. The V2 OpenVINO
            // entry point takes the same string map, and going through the generic one
            // keeps this crate from binding a struct whose layout Intel may change.
            Self::OpenVino {
                device_type,
                num_of_threads,
                num_streams,
            } => {
                let mut opts = vec![("device_type".to_owned(), device_type.clone())];
                if let Some(n) = num_of_threads {
                    opts.push(("num_of_threads".to_owned(), n.to_string()));
                }
                if let Some(n) = num_streams {
                    opts.push(("num_streams".to_owned(), n.to_string()));
                }
                append_by_name(api, options, "OpenVINO", &opts)
            }
            Self::Custom { name, options: kv } => append_by_name(api, options, name, kv),

            // TensorRT exposes a factory that returns an options object *already
            // populated with ONNX Runtime's defaults*. Going through it means this
            // crate never hardcodes a default, which matters because no NVIDIA
            // hardware is available to test against.
            Self::TensorRt { device_id } => append_via_factory!(
                api,
                options,
                &[("device_id".to_owned(), device_id.to_string())],
                CreateTensorRTProviderOptions,
                UpdateTensorRTProviderOptions,
                ReleaseTensorRTProviderOptions,
                SessionOptionsAppendExecutionProvider_TensorRT_V2
            ),
            // MIGraphX is the exception: no factory exists, so the struct has to
            // be filled here. Note `migraphx_mem_limit` — zeroing it would mean
            // *zero bytes*, not "unlimited", so the defaults are set explicitly.
            Self::MiGraphx { device_id } => {
                let append = ort_fn!(api, SessionOptionsAppendExecutionProvider_MIGraphX);
                let migraphx_options = sys::OrtMIGraphXProviderOptions {
                    device_id: *device_id,
                    migraphx_fp16_enable: 0,
                    migraphx_fp8_enable: 0,
                    migraphx_int8_enable: 0,
                    migraphx_use_native_calibration_table: 0,
                    migraphx_int8_calibration_table_name: ptr::null(),
                    migraphx_save_compiled_model: 0,
                    migraphx_save_model_path: ptr::null(),
                    migraphx_load_compiled_model: 0,
                    migraphx_load_model_path: ptr::null(),
                    migraphx_exhaustive_tune: false,
                    migraphx_mem_limit: usize::MAX,
                    migraphx_arena_extend_strategy: 0,
                };
                // SAFETY: `options` is a live session-options object and
                // `migraphx_options` outlives the call, which only reads it.
                unsafe { api.check(append(options, &migraphx_options)) }
            }
        }
    }
}

/// Parses a provider from configuration.
///
/// Case-insensitive, with an optional `:device_id` suffix for the GPU providers:
///
/// | Input | Result |
/// |---|---|
/// | `cpu` | [`ExecutionProvider::Cpu`] |
/// | `coreml`, `core_ml` | CoreML, all compute units, ML Program format |
/// | `coreml:cpu_and_neural_engine` | CoreML, restricted to CPU and Neural Engine |
/// | `coreml:cpu_and_neural_engine:neuralnetwork` | the older format, for an old host |
/// | `xnnpack` | XNNPACK |
/// | `tensorrt`, `trt:0`, `tensorrt:1` | NVIDIA, on device 0 or 1 |
/// | `migraphx`, `migraphx:0` | AMD |
///
/// Anything else becomes [`ExecutionProvider::Custom`] with no options, so a
/// provider this enum does not model can still be named in configuration.
impl FromStr for ExecutionProvider {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let (head, tail) = match s.split_once(':') {
            Some((h, t)) => (h, Some(t)),
            None => (s, None),
        };
        let head = head.trim().to_ascii_lowercase();

        let device_id = || -> Result<i32> {
            match tail {
                None => Ok(0),
                Some(t) => t.trim().parse::<i32>().map_err(|_| Error::InvalidProvider {
                    input: s.to_owned(),
                    reason: format!("expected a device id after ':', got {t:?}"),
                }),
            }
        };

        match head.as_str() {
            "cpu" => Ok(Self::Cpu),
            "coreml" | "core_ml" => {
                // Two optional suffixes, either order, because "which hardware" and
                // "which format" are independent choices and a configuration should not
                // have to name one to reach the other.
                let mut compute_units = CoreMlComputeUnits::All;
                let mut model_format = CoreMlModelFormat::default();
                for part in tail.into_iter().flat_map(|t| t.split(':')) {
                    match part.trim().to_ascii_lowercase().as_str() {
                        "" => continue,
                        "all" => compute_units = CoreMlComputeUnits::All,
                        "cpu_and_gpu" | "cpuandgpu" => {
                            compute_units = CoreMlComputeUnits::CpuAndGpu;
                        }
                        "cpu_and_neural_engine" | "cpuandneuralengine" => {
                            compute_units = CoreMlComputeUnits::CpuAndNeuralEngine;
                        }
                        "cpu_only" | "cpuonly" => compute_units = CoreMlComputeUnits::CpuOnly,
                        "mlprogram" | "ml_program" => {
                            model_format = CoreMlModelFormat::MlProgram;
                        }
                        "neuralnetwork" | "neural_network" => {
                            model_format = CoreMlModelFormat::NeuralNetwork;
                        }
                        other => {
                            return Err(Error::InvalidProvider {
                                input: s.to_owned(),
                                reason: format!(
                                    "unknown CoreML option {other:?}; expected compute \
                                     units (all, cpu_and_gpu, cpu_and_neural_engine, \
                                     cpu_only) or a model format (mlprogram, \
                                     neuralnetwork)"
                                ),
                            });
                        }
                    }
                }
                Ok(Self::CoreMl {
                    compute_units,
                    model_format,
                })
            }
            "openvino" | "open_vino" => {
                // Tokens in any order, like CoreML: a bare token is the device, and
                // `key=value` is an option. The device is uppercased because OpenVINO's
                // own names are, and a lowercase `cpu` in a configuration should not be
                // a different thing from `CPU`.
                //
                // A multi-device form contains a colon of its own -- `AUTO:GPU,CPU` --
                // so the mode and its device list arrive as two tokens and are rejoined.
                let mut device: Option<String> = None;
                let mut num_of_threads = None;
                let mut num_streams = None;
                let bad = |reason: String| Error::InvalidProvider {
                    input: s.to_owned(),
                    reason,
                };
                for part in tail.into_iter().flat_map(|t| t.split(':')) {
                    let part = part.trim();
                    if part.is_empty() {
                        continue;
                    }
                    match part.split_once('=') {
                        Some((key, value)) => {
                            // The key is checked before the value is parsed, so an
                            // option this does not model is reported as unknown rather
                            // than as a number that would not parse.
                            let key = key.trim().to_ascii_lowercase();
                            let slot = match key.as_str() {
                                "threads" | "num_of_threads" => &mut num_of_threads,
                                "streams" | "num_streams" => &mut num_streams,
                                other => {
                                    return Err(bad(format!(
                                        "unknown OpenVINO option {other:?}; expected \
                                         threads or streams. Everything else OpenVINO \
                                         takes goes through load_config, which a provider \
                                         string cannot carry"
                                    )));
                                }
                            };
                            *slot = Some(value.trim().parse::<u32>().map_err(|_| {
                                bad(format!(
                                    "expected a positive number for OpenVINO {key:?}, \
                                     got {value:?}"
                                ))
                            })?);
                        }
                        // A device list following a multi-device mode, rejoined onto it.
                        None if device
                            .as_deref()
                            .is_some_and(|d| matches!(d, "AUTO" | "HETERO" | "MULTI")) =>
                        {
                            let mode = device.take().unwrap_or_default();
                            device = Some(format!("{mode}:{}", part.to_ascii_uppercase()));
                        }
                        None if device.is_some() => {
                            return Err(bad(format!(
                                "two device types given for OpenVINO, {:?} and {part:?}; \
                                 name one",
                                device.unwrap_or_default()
                            )));
                        }
                        None => device = Some(part.to_ascii_uppercase()),
                    }
                }
                if num_of_threads == Some(0) || num_streams == Some(0) {
                    return Err(bad(
                        "OpenVINO threads and streams must be positive; leave the option \
                         out for its own default"
                            .to_owned(),
                    ));
                }
                Ok(Self::OpenVino {
                    // CPU because that is what this provider is reached for on a server;
                    // a GPU or NPU has to be named.
                    device_type: device.unwrap_or_else(|| "CPU".to_owned()),
                    num_of_threads,
                    num_streams,
                })
            }
            "xnnpack" => Ok(Self::Xnnpack {
                intra_op_threads: match tail {
                    None => None,
                    Some(t) => {
                        Some(
                            t.trim()
                                .parse::<u32>()
                                .map_err(|_| Error::InvalidProvider {
                                    input: s.to_owned(),
                                    reason: format!("expected a thread count after ':', got {t:?}"),
                                })?,
                        )
                    }
                },
            }),
            // Deliberately rejected rather than passed through: CUDA and ROCm
            // have dedicated appenders, so treating them as `Custom` would send
            // them down the generic path and fail confusingly. Naming the
            // supported option is more useful than either.
            "cuda" => Err(Error::InvalidProvider {
                input: s.to_owned(),
                reason: "the CUDA provider is not offered; use \"tensorrt\" for NVIDIA".to_owned(),
            }),
            "tensorrt" | "trt" => Ok(Self::TensorRt {
                device_id: device_id()?,
            }),
            "rocm" => Err(Error::InvalidProvider {
                input: s.to_owned(),
                reason: "the ROCm provider is not offered; use \"migraphx\" for AMD".to_owned(),
            }),
            "migraphx" => Ok(Self::MiGraphx {
                device_id: device_id()?,
            }),
            _ => Ok(Self::Custom {
                name: s.trim().to_owned(),
                options: Vec::new(),
            }),
        }
    }
}

impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoreMl {
                compute_units,
                model_format,
            } => write!(
                f,
                "CoreML({}, {})",
                compute_units.option_value(),
                model_format.option_value()
            ),
            Self::Xnnpack {
                intra_op_threads: Some(n),
            } => write!(f, "XNNPACK(threads={n})"),
            Self::OpenVino {
                device_type,
                num_of_threads,
                num_streams,
            } => {
                write!(f, "OpenVINO({device_type}")?;
                if let Some(n) = num_of_threads {
                    write!(f, ", threads={n}")?;
                }
                if let Some(n) = num_streams {
                    write!(f, ", streams={n}")?;
                }
                f.write_str(")")
            }
            Self::TensorRt { device_id } | Self::MiGraphx { device_id } => {
                write!(f, "{}(device={device_id})", self.name())
            }
            _ => f.write_str(self.name()),
        }
    }
}

impl Api {
    /// The execution providers compiled into the loaded ONNX Runtime.
    ///
    /// Names are the long registry form, for example `"CPUExecutionProvider"`.
    /// This is the only trustworthy answer to "can I use TensorRT here?", because the
    /// library is resolved at run time.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let api = onnxrt_rs::Api::load()?;
    /// println!("{}", api.available_providers()?.join(", "));
    /// # Ok(())
    /// # }
    /// ```
    pub fn available_providers(&self) -> Result<Vec<String>> {
        let get = ort_fn!(*self, GetAvailableProviders);
        let mut raw: *mut *mut c_char = ptr::null_mut();
        let mut len: c_int = 0;
        // SAFETY: both are valid out-parameters. On success ONNX Runtime writes an
        // array of `len` C strings, freed by `ReleaseAvailableProviders` below.
        unsafe { self.check(get(&mut raw, &mut len))? };

        let count = len.max(0) as usize;
        let mut providers = Vec::with_capacity(count);
        for i in 0..count {
            // SAFETY: `raw` holds `count` valid C strings, per the call above.
            let name = unsafe { CStr::from_ptr(*raw.add(i)) };
            providers.push(name.to_string_lossy().into_owned());
        }

        if let Some(release) = self.raw().ReleaseAvailableProviders {
            // SAFETY: `raw`/`len` are exactly what `GetAvailableProviders` wrote.
            // The returned status is ignored: a failure to free is not
            // actionable, and the names have already been copied out.
            unsafe { release(raw, len) };
        }

        Ok(providers)
    }
}

/// The accelerator features this build was compiled with.
///
/// Used in error messages so a misconfigured deployment is told what this binary
/// *can* do, not merely what it cannot.
pub fn enabled_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "cpu") {
        features.push("cpu");
    }
    if cfg!(feature = "coreml") {
        features.push("coreml");
    }
    if cfg!(feature = "xnnpack") {
        features.push("xnnpack");
    }
    if cfg!(feature = "tensorrt") {
        features.push("tensorrt");
    }
    if cfg!(feature = "migraphx") {
        features.push("migraphx");
    }
    features
}

/// Folds a provider name to a comparable form, so `"CoreML"`,
/// `"CoreMLExecutionProvider"` and `"coreml"` all compare equal.
fn normalized_provider_name(name: &str) -> String {
    name.trim()
        .trim_end_matches("ExecutionProvider")
        .to_ascii_lowercase()
}

/// Appends a provider through the generic string-keyed entry point.
fn append_by_name(
    api: Api,
    options: *mut sys::OrtSessionOptions,
    name: &str,
    kv: &[(String, String)],
) -> Result<()> {
    let append = ort_fn!(api, SessionOptionsAppendExecutionProvider);
    let c_name = CString::new(name).map_err(|_| Error::InteriorNul {
        context: "execution provider name",
    })?;
    let (keys, values) = c_pairs(kv)?;
    let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
    let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();

    // An empty option list passes null rather than a dangling pointer from an
    // empty Vec, even though ONNX Runtime should not read it when num_keys is 0.
    let (kp, vp) = if key_ptrs.is_empty() {
        (ptr::null(), ptr::null())
    } else {
        (key_ptrs.as_ptr(), value_ptrs.as_ptr())
    };

    // SAFETY: `options` is live; the name, keys and values are valid, NUL-
    // terminated, and outlive the call, which only reads them.
    unsafe { api.check(append(options, c_name.as_ptr(), kp, vp, key_ptrs.len())) }
}

/// Converts option pairs into NUL-terminated strings, keeping them alive for the
/// caller to borrow.
fn c_pairs(kv: &[(String, String)]) -> Result<(Vec<CString>, Vec<CString>)> {
    let mut keys = Vec::with_capacity(kv.len());
    let mut values = Vec::with_capacity(kv.len());
    for (k, v) in kv {
        keys.push(CString::new(k.as_str()).map_err(|_| Error::InteriorNul {
            context: "execution provider option key",
        })?);
        values.push(CString::new(v.as_str()).map_err(|_| Error::InteriorNul {
            context: "execution provider option value",
        })?);
    }
    Ok((keys, values))
}
