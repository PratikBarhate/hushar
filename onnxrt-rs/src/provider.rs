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
//! An EP is compiled *into* `libonnxruntime`, so which ones can be used is decided
//! by the build that is shipped, not by any Cargo feature here:
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
//! The library is resolved at run time, so the only trustworthy answer to "is
//! TensorRT available?" comes from [`Api::available_providers`].
//! [`ExecutionProvider::append`] checks against that list first, so requesting a
//! provider the loaded library lacks produces [`Error::ProviderUnavailable`],
//! naming what *is* available.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::str::FromStr;

use crate::api::{Api, ort_fn};
use crate::error::{Error, Result};
use crate::sys;

/// Creates a provider-options object through ONNX Runtime's own factory, applies the
/// string options to it, registers it, and releases it on every path.
///
/// The factory hands back an object already populated with ONNX Runtime's defaults,
/// so only the options passed here are overridden. ONNX Runtime copies what it needs
/// out of the options object during the append, so releasing it afterwards is correct.
macro_rules! append_via_factory {
    ($api:expr, $options:expr, $kv:expr, $create:ident, $update:ident, $release:ident, $append:ident) => {{
        let api: Api = $api;
        let create = ort_fn!(api, $create);
        let update = ort_fn!(api, $update);
        let append = ort_fn!(api, $append);

        let mut provider_options = ptr::null_mut();
        unsafe { api.check(create(&mut provider_options))? };

        let outcome = (|| -> Result<()> {
            let (keys, values) = c_pairs($kv)?;
            if !keys.is_empty() {
                let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
                let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
                unsafe {
                    api.check(update(
                        provider_options,
                        key_ptrs.as_ptr(),
                        value_ptrs.as_ptr(),
                        key_ptrs.len(),
                    ))?
                };
            }
            unsafe { api.check(append($options, provider_options)) }
        })();

        if let Some(release) = api.raw().$release {
            unsafe { release(provider_options) };
        }
        outcome
    }};
}

/// Which compute units the CoreML provider may use.
///
/// Maps to CoreML's `MLComputeUnits` provider option.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CoreMlComputeUnits {
    /// CPU, GPU and Neural Engine. CoreML decides placement.
    #[default]
    All,
    /// CPU and GPU only.
    CpuAndGpu,
    /// CPU and Neural Engine only.
    CpuAndNeuralEngine,
    /// CPU only.
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
/// `MatMul` and hands those nodes back to the CPU. `MlProgram` needs macOS 12 or
/// iOS 15; `NeuralNetwork` is the older format, for a host below that.
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
    /// even when another provider is registered.
    Cpu,

    /// Apple CoreML — Apple Silicon GPU and Neural Engine.
    ///
    /// # Fields
    ///
    /// - `compute_units` — which compute units CoreML may use.
    /// - `model_format` — which CoreML format the graph is converted into.
    CoreMl {
        compute_units: CoreMlComputeUnits,
        model_format: CoreMlModelFormat,
    },

    /// XNNPACK — optimised CPU kernels.
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
    ///
    /// Note the coverage tradeoff: TensorRT does not implement every ONNX
    /// operator, and nodes it cannot take fall back to **CPU**, which means
    /// host-device copies mid-graph.
    TensorRt {
        /// CUDA device ordinal.
        device_id: i32,
    },

    /// AMD, through MIGraphX.
    ///
    /// MIGraphX is AMD's graph-optimising inference compiler, layered over ROCm.
    /// The same fallback-to-CPU caveat as [`Self::TensorRt`] applies.
    MiGraphx {
        /// HIP device ordinal.
        device_id: i32,
    },

    /// Intel's OpenVINO, on Intel CPU, integrated or discrete GPU, or NPU.
    ///
    /// Not in any prebuilt ONNX Runtime archive: it needs a source build with
    /// `--use_openvino`, and OpenVINO itself installed.
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
    /// An escape hatch for providers this enum does not model — QNN, WebGPU,
    /// VitisAI — and for options it does not model.
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
    /// Always the long `…ExecutionProvider` form, which is what the registry uses,
    /// even though the append call accepts the short form too.
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
    /// [`Self::Custom`] is not gated, and so returns `None`.
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
    /// The CPU provider is implicit and is not always listed in the registry, so it
    /// short-circuits before the availability check. MIGraphX has no options factory,
    /// so its raw struct is filled here with ONNX Runtime's defaults spelled out —
    /// note that a zero `migraphx_mem_limit` would mean *zero bytes* rather than
    /// unlimited.
    ///
    /// # Errors
    ///
    /// [`Error::ProviderNotEnabled`] when this crate was compiled without the
    /// provider's Cargo feature, and [`Error::ProviderUnavailable`] — naming what is
    /// available — when the loaded `libonnxruntime` was not built with it. The two are
    /// checked separately because the feature says the deployment intends to use the
    /// hardware and the runtime list says the library can.
    pub(crate) fn append(&self, api: Api, options: *mut sys::OrtSessionOptions) -> Result<()> {
        if !self.is_enabled() {
            return Err(Error::ProviderNotEnabled {
                provider: self.name().to_owned(),
                feature: self.required_feature().unwrap_or("<none>"),
                enabled: enabled_features().join(", "),
            });
        }

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

            Self::TensorRt { device_id } => append_via_factory!(
                api,
                options,
                &[("device_id".to_owned(), device_id.to_string())],
                CreateTensorRTProviderOptions,
                UpdateTensorRTProviderOptions,
                ReleaseTensorRTProviderOptions,
                SessionOptionsAppendExecutionProvider_TensorRT_V2
            ),
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
/// | `openvino`, `openvino:gpu:threads=8` | Intel, on the named device |
/// | `tensorrt`, `trt:0`, `tensorrt:1` | NVIDIA, on device 0 or 1 |
/// | `migraphx`, `migraphx:0` | AMD |
///
/// The two CoreML suffixes — compute units and model format — are independent, so
/// either may be omitted and they may appear in either order.
///
/// OpenVINO takes its suffixes in any order too: a bare token is the device type,
/// uppercased, and `key=value` sets `threads` or `streams`, which must be positive.
/// The device type defaults to `CPU`. A multi-device form carries a colon of its own,
/// as in `openvino:auto:gpu,cpu`, and the mode and its device list are rejoined.
///
/// `cuda` and `rocm` are rejected rather than passed through, because they have
/// dedicated appenders that the generic path would not reach; use `tensorrt` and
/// `migraphx`. Anything else becomes [`ExecutionProvider::Custom`] with no options,
/// so a provider this enum does not model can still be named in configuration.
///
/// # Errors
///
/// [`Error::InvalidProvider`], naming what was expected, for an unparseable device
/// id, thread or stream count, an unknown CoreML or OpenVINO option, two device
/// types for OpenVINO, or one of the two rejected providers.
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
    /// Each name is copied out before `ReleaseAvailableProviders` frees the array; a
    /// failure to free is ignored, since it is not actionable once the names are copied.
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
        unsafe { self.check(get(&mut raw, &mut len))? };

        let count = len.max(0) as usize;
        let mut providers = Vec::with_capacity(count);
        for i in 0..count {
            let name = unsafe { CStr::from_ptr(*raw.add(i)) };
            providers.push(name.to_string_lossy().into_owned());
        }

        if let Some(release) = self.raw().ReleaseAvailableProviders {
            unsafe { release(raw, len) };
        }

        Ok(providers)
    }
}

/// The accelerator features this build was compiled with.
///
/// Used in error messages, so a misconfigured deployment is told what this binary
/// can do as well as what it cannot.
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
///
/// An empty option list passes null rather than the dangling pointer an empty `Vec`
/// would yield.
///
/// # Errors
///
/// [`Error::InteriorNul`] if the provider name or any option contains a NUL byte.
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

    let (kp, vp) = if key_ptrs.is_empty() {
        (ptr::null(), ptr::null())
    } else {
        (key_ptrs.as_ptr(), value_ptrs.as_ptr())
    };

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
