// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Errors produced by the safe ONNX Runtime front end.

use std::ffi::CStr;
use std::fmt;

use crate::sys;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// An `OrtErrorCode` returned by the C API.
///
/// ONNX Runtime may add codes in future releases, so unrecognised values are
/// preserved in [`ErrorCode::Other`] rather than being collapsed into a
/// catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Fail,
    InvalidArgument,
    NoSuchFile,
    NoModel,
    EngineError,
    RuntimeException,
    InvalidProtobuf,
    ModelLoaded,
    NotImplemented,
    InvalidGraph,
    EpFail,
    ModelLoadCanceled,
    ModelRequiresCompilation,
    NotFound,
    DeviceReset,
    Other(u32),
}

impl ErrorCode {
    fn from_raw(code: sys::OrtErrorCode) -> Self {
        match code {
            sys::OrtErrorCode_ORT_FAIL => Self::Fail,
            sys::OrtErrorCode_ORT_INVALID_ARGUMENT => Self::InvalidArgument,
            sys::OrtErrorCode_ORT_NO_SUCHFILE => Self::NoSuchFile,
            sys::OrtErrorCode_ORT_NO_MODEL => Self::NoModel,
            sys::OrtErrorCode_ORT_ENGINE_ERROR => Self::EngineError,
            sys::OrtErrorCode_ORT_RUNTIME_EXCEPTION => Self::RuntimeException,
            sys::OrtErrorCode_ORT_INVALID_PROTOBUF => Self::InvalidProtobuf,
            sys::OrtErrorCode_ORT_MODEL_LOADED => Self::ModelLoaded,
            sys::OrtErrorCode_ORT_NOT_IMPLEMENTED => Self::NotImplemented,
            sys::OrtErrorCode_ORT_INVALID_GRAPH => Self::InvalidGraph,
            sys::OrtErrorCode_ORT_EP_FAIL => Self::EpFail,
            sys::OrtErrorCode_ORT_MODEL_LOAD_CANCELED => Self::ModelLoadCanceled,
            sys::OrtErrorCode_ORT_MODEL_REQUIRES_COMPILATION => Self::ModelRequiresCompilation,
            sys::OrtErrorCode_ORT_NOT_FOUND => Self::NotFound,
            sys::OrtErrorCode_ORT_DEVICE_RESET => Self::DeviceReset,
            other => Self::Other(other),
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Fail => "FAIL",
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::NoSuchFile => "NO_SUCHFILE",
            Self::NoModel => "NO_MODEL",
            Self::EngineError => "ENGINE_ERROR",
            Self::RuntimeException => "RUNTIME_EXCEPTION",
            Self::InvalidProtobuf => "INVALID_PROTOBUF",
            Self::ModelLoaded => "MODEL_LOADED",
            Self::NotImplemented => "NOT_IMPLEMENTED",
            Self::InvalidGraph => "INVALID_GRAPH",
            Self::EpFail => "EP_FAIL",
            Self::ModelLoadCanceled => "MODEL_LOAD_CANCELED",
            Self::ModelRequiresCompilation => "MODEL_REQUIRES_COMPILATION",
            Self::NotFound => "NOT_FOUND",
            Self::DeviceReset => "DEVICE_RESET",
            Self::Other(c) => return write!(f, "UNKNOWN({c})"),
        };
        f.write_str(s)
    }
}

/// Anything that can go wrong when driving ONNX Runtime.
/// Render a [`libloading::Error`] together with its whole source chain.
///
/// `libloading`'s own `Display` for a failed `dlopen` is the bare string
/// "dlopen failed". The text from `dlerror` — the part that actually says *why*,
/// whether that is a missing transitive dependency, a `glibc` older than the one
/// the runtime was built against, or an architecture mismatch — sits one level
/// down in [`std::error::Error::source`]. Formatting only the top-level error
/// hands an operator a message with no diagnostic content, so walk the chain.
fn describe_load_error(error: &libloading::Error) -> String {
    use std::error::Error as _;

    let mut text = error.to_string();
    let mut cause = error.source();
    while let Some(current) = cause {
        text.push_str(": ");
        text.push_str(&current.to_string());
        cause = current.source();
    }
    text
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The shared library could not be opened.
    #[error(
        "could not load the ONNX Runtime shared library ({path}): {}",
        describe_load_error(source)
    )]
    LibraryLoad {
        path: String,
        #[source]
        source: libloading::Error,
    },

    /// The library opened but did not export `OrtGetApiBase`.
    #[error("`OrtGetApiBase` is missing from the ONNX Runtime shared library: {0}")]
    MissingEntryPoint(#[source] libloading::Error),

    /// The loaded runtime is older than the header these bindings were built
    /// from, so it cannot supply an `OrtApi` of the requested version.
    #[error(
        "the loaded ONNX Runtime does not support API version {requested}; \
         the library reports version \"{library_version}\". Upgrade the ONNX \
         Runtime shared library, or regenerate the bindings against its headers."
    )]
    UnsupportedApiVersion {
        requested: u32,
        library_version: String,
    },

    /// A required `OrtApi` function pointer was null.
    #[error("the ONNX Runtime API table has no `{0}` entry")]
    MissingApiFunction(&'static str),

    /// The C API returned a non-null `OrtStatus`.
    #[error("ONNX Runtime error [{code}]: {message}")]
    Ort { code: ErrorCode, message: String },

    /// A string handed to the C API contained an interior NUL byte.
    #[error("`{context}` contains an interior NUL byte and cannot be passed to C")]
    InteriorNul { context: &'static str },

    /// The caller's buffer length does not match the shape it claims to have.
    #[error("shape {shape:?} describes {expected} elements but {actual} were supplied")]
    ShapeMismatch {
        shape: Vec<i64>,
        expected: usize,
        actual: usize,
    },

    /// A tensor held a different element type than the one requested.
    #[error("tensor element type mismatch: requested {requested}, tensor holds {actual}")]
    ElementTypeMismatch {
        requested: &'static str,
        actual: u32,
    },

    /// The model input or output is not a tensor, so it has no element type.
    ///
    /// ONNX also allows sequences and maps. hushar serves tensors, so this is
    /// reported rather than worked around.
    #[error("model {kind} {index} is not a tensor, so it has no element type")]
    NotATensor { kind: &'static str, index: usize },

    /// The model uses an element type this crate does not model.
    ///
    /// ONNX defines 8-bit and 4-bit float formats, sub-byte integers and complex
    /// numbers that have no Rust equivalent here. Reported rather than guessed at,
    /// because reinterpreting the bytes as a type of the same width would be
    /// silently wrong.
    #[error("ONNX element type {raw} is not supported by this crate")]
    UnsupportedDataType { raw: u32 },

    /// A string tensor's offsets did not describe a valid layout.
    ///
    /// The runtime should never produce this; it is checked because slicing on a
    /// bad offset would panic inside a request handler.
    #[error(
        "string tensor element {index} spans {start}..{end}, which is not within \
         its {total}-byte payload"
    )]
    MalformedStringTensor {
        index: usize,
        start: usize,
        end: usize,
        total: usize,
    },

    /// An input was supplied that the model does not declare.
    #[error("the model has no input named {name:?}; it declares: {declared}")]
    UnknownInput { name: String, declared: String },

    /// A model input was supplied more than once, or not at all.
    #[error("input {name:?} was supplied {count} times; each must be supplied exactly once")]
    InputNotSuppliedOnce { name: String, count: usize },

    /// The runtime reported a shape containing a symbolic or negative dimension
    /// where a concrete one was required.
    #[error("tensor has non-concrete dimension {dim} in shape {shape:?}")]
    NonConcreteDimension { shape: Vec<i64>, dim: i64 },

    /// The model has no input or output at the requested index.
    #[error("{kind} index {index} is out of range (model has {count})")]
    IndexOutOfRange {
        kind: &'static str,
        index: usize,
        count: usize,
    },

    /// The requested execution provider is not compiled into the loaded library.
    ///
    /// Providers are built *into* `libonnxruntime`, so this is a property of the
    /// binary that was loaded, not of this crate. Either install a build that
    /// includes the provider, or choose one from `available`.
    #[error(
        "execution provider \"{provider}\" is not available in the loaded ONNX \
         Runtime; it was built with: {available}"
    )]
    ProviderUnavailable { provider: String, available: String },

    /// An execution provider could not be parsed from configuration.
    #[error("cannot parse execution provider from {input:?}: {reason}")]
    InvalidProvider { input: String, reason: String },

    /// This build was not compiled with the requested provider's feature.
    ///
    /// Distinct from [`Self::ProviderUnavailable`]: that one means the loaded
    /// `libonnxruntime` lacks the provider, this one means the binary was not
    /// built to use it. Rebuild with the named Cargo feature, or with
    /// `all-providers`.
    #[error(
        "execution provider \"{provider}\" needs the `{feature}` feature, which \
         this build does not enable; it was built with: {enabled}"
    )]
    ProviderNotEnabled {
        provider: String,
        feature: &'static str,
        enabled: String,
    },
}

impl Error {
    /// Builds an [`Error::Ort`] from a non-null `OrtStatus`, releasing the
    /// status before returning.
    ///
    /// # Safety
    ///
    /// `status` must be a non-null `OrtStatus` obtained from `api`, and must not
    /// be used again after this call: ownership is transferred here.
    pub(crate) unsafe fn from_status(api: &sys::OrtApi, status: sys::OrtStatusPtr) -> Self {
        // SAFETY: the caller guarantees `status` came from this API table and is
        // non-null. `GetErrorCode`/`GetErrorMessage` are infallible accessors,
        // and the message buffer is owned by the status, so it must be copied
        // before `ReleaseStatus` runs.
        let (code, message) = unsafe {
            let code = api
                .GetErrorCode
                .map(|f| f(status))
                .unwrap_or(sys::OrtErrorCode_ORT_FAIL);
            let message = api
                .GetErrorMessage
                .map(|f| f(status))
                .filter(|p| !p.is_null())
                .map(|p| CStr::from_ptr(p).to_string_lossy().into_owned())
                .unwrap_or_else(|| "<no message>".to_owned());
            if let Some(release) = api.ReleaseStatus {
                release(status);
            }
            (code, message)
        };

        Self::Ort {
            code: ErrorCode::from_raw(code),
            message,
        }
    }
}
