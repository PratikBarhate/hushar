// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Errors produced by the safe Neuron runtime front end.

use std::fmt;

use crate::sys;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// An `NRT_STATUS` returned by `libnrt`.
///
/// The variants worth branching on are called out individually; the rest are
/// preserved in [`Status::Other`] with the raw code so nothing is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Generic failure with no more specific code.
    Failure,
    /// Bad NEFF, bad instruction, or an input tensor that does not match the
    /// model's expectations.
    Invalid,
    /// A handle passed to the API was not valid.
    InvalidHandle,
    /// Could not allocate a resource for the request.
    Resource,
    /// The operation timed out.
    Timeout,
    /// Hardware failure.
    HardwareError,
    /// The execution input queue is full.
    QueueFull,
    /// Not enough NeuronCores free to load the NEFF.
    NotEnoughNeuronCores,
    /// The NEFF was built for an unsupported runtime version.
    UnsupportedNeffVersion,
    /// An API was called before `nrt_init`.
    Uninitialized,
    /// An API was called after `nrt_close`.
    Closed,
    /// Invalid input submitted to execute.
    ExecBadInput,
    /// Execution finished but produced NaN.
    ExecCompletedWithNumericalError,
    /// Execution finished with a logical or physical error.
    ExecCompletedWithError,
    /// The NeuronCore is in use by another model or process.
    ExecNeuronCoreBusy,
    /// An indirect copy or embedding update went out of bounds.
    ExecOutOfBounds,
    /// A status this crate does not name explicitly.
    Other(u32),
}

impl Status {
    /// Maps a raw `NRT_STATUS`, returning `None` for `NRT_SUCCESS`.
    pub fn from_raw(status: sys::NRT_STATUS) -> Option<Self> {
        Some(match status {
            sys::NRT_STATUS_NRT_SUCCESS => return None,
            sys::NRT_STATUS_NRT_FAILURE => Self::Failure,
            sys::NRT_STATUS_NRT_INVALID => Self::Invalid,
            sys::NRT_STATUS_NRT_INVALID_HANDLE => Self::InvalidHandle,
            sys::NRT_STATUS_NRT_RESOURCE => Self::Resource,
            sys::NRT_STATUS_NRT_TIMEOUT => Self::Timeout,
            sys::NRT_STATUS_NRT_HW_ERROR => Self::HardwareError,
            sys::NRT_STATUS_NRT_QUEUE_FULL => Self::QueueFull,
            sys::NRT_STATUS_NRT_LOAD_NOT_ENOUGH_NC => Self::NotEnoughNeuronCores,
            sys::NRT_STATUS_NRT_UNSUPPORTED_NEFF_VERSION => Self::UnsupportedNeffVersion,
            sys::NRT_STATUS_NRT_UNINITIALIZED => Self::Uninitialized,
            sys::NRT_STATUS_NRT_CLOSED => Self::Closed,
            sys::NRT_STATUS_NRT_EXEC_BAD_INPUT => Self::ExecBadInput,
            sys::NRT_STATUS_NRT_EXEC_COMPLETED_WITH_NUM_ERR => {
                Self::ExecCompletedWithNumericalError
            }
            sys::NRT_STATUS_NRT_EXEC_COMPLETED_WITH_ERR => Self::ExecCompletedWithError,
            sys::NRT_STATUS_NRT_EXEC_NC_BUSY => Self::ExecNeuronCoreBusy,
            sys::NRT_STATUS_NRT_EXEC_OOB => Self::ExecOutOfBounds,
            other => Self::Other(other),
        })
    }

    /// The raw `NRT_STATUS` value.
    pub fn as_raw(self) -> sys::NRT_STATUS {
        match self {
            Self::Failure => sys::NRT_STATUS_NRT_FAILURE,
            Self::Invalid => sys::NRT_STATUS_NRT_INVALID,
            Self::InvalidHandle => sys::NRT_STATUS_NRT_INVALID_HANDLE,
            Self::Resource => sys::NRT_STATUS_NRT_RESOURCE,
            Self::Timeout => sys::NRT_STATUS_NRT_TIMEOUT,
            Self::HardwareError => sys::NRT_STATUS_NRT_HW_ERROR,
            Self::QueueFull => sys::NRT_STATUS_NRT_QUEUE_FULL,
            Self::NotEnoughNeuronCores => sys::NRT_STATUS_NRT_LOAD_NOT_ENOUGH_NC,
            Self::UnsupportedNeffVersion => sys::NRT_STATUS_NRT_UNSUPPORTED_NEFF_VERSION,
            Self::Uninitialized => sys::NRT_STATUS_NRT_UNINITIALIZED,
            Self::Closed => sys::NRT_STATUS_NRT_CLOSED,
            Self::ExecBadInput => sys::NRT_STATUS_NRT_EXEC_BAD_INPUT,
            Self::ExecCompletedWithNumericalError => {
                sys::NRT_STATUS_NRT_EXEC_COMPLETED_WITH_NUM_ERR
            }
            Self::ExecCompletedWithError => sys::NRT_STATUS_NRT_EXEC_COMPLETED_WITH_ERR,
            Self::ExecNeuronCoreBusy => sys::NRT_STATUS_NRT_EXEC_NC_BUSY,
            Self::ExecOutOfBounds => sys::NRT_STATUS_NRT_EXEC_OOB,
            Self::Other(code) => code,
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Failure => "NRT_FAILURE",
            Self::Invalid => "NRT_INVALID",
            Self::InvalidHandle => "NRT_INVALID_HANDLE",
            Self::Resource => "NRT_RESOURCE",
            Self::Timeout => "NRT_TIMEOUT",
            Self::HardwareError => "NRT_HW_ERROR",
            Self::QueueFull => "NRT_QUEUE_FULL",
            Self::NotEnoughNeuronCores => "NRT_LOAD_NOT_ENOUGH_NC",
            Self::UnsupportedNeffVersion => "NRT_UNSUPPORTED_NEFF_VERSION",
            Self::Uninitialized => "NRT_UNINITIALIZED",
            Self::Closed => "NRT_CLOSED",
            Self::ExecBadInput => "NRT_EXEC_BAD_INPUT",
            Self::ExecCompletedWithNumericalError => "NRT_EXEC_COMPLETED_WITH_NUM_ERR",
            Self::ExecCompletedWithError => "NRT_EXEC_COMPLETED_WITH_ERR",
            Self::ExecNeuronCoreBusy => "NRT_EXEC_NC_BUSY",
            Self::ExecOutOfBounds => "NRT_EXEC_OOB",
            Self::Other(code) => return write!(f, "NRT_STATUS({code})"),
        };
        f.write_str(s)
    }
}

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

/// Anything that can go wrong when driving the Neuron runtime.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The Neuron runtime is only available on Linux.
    #[error(
        "the AWS Neuron runtime is only available on Linux (this binary was built \
         for {os}/{arch}); Neuron code paths cannot run here"
    )]
    UnsupportedPlatform {
        os: &'static str,
        arch: &'static str,
    },

    /// `libnrt` could not be opened.
    #[error(
        "could not load the Neuron runtime shared library ({path}): {}. \
         Install aws-neuronx-runtime-lib, or set NRT_DYLIB_PATH.",
        describe_load_error(source)
    )]
    LibraryLoad {
        path: String,
        #[source]
        source: libloading::Error,
    },

    /// A required symbol was absent from `libnrt`.
    #[error("`{symbol}` is missing from the Neuron runtime shared library: {source}")]
    MissingSymbol {
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },

    /// `libnrt` returned a failure status.
    ///
    /// `detail` is built by the crate's internal `Symbols::check`, which folds the
    /// status name and `nrt_get_status_as_str` text into one string. For most
    /// statuses `nrt_get_status_as_str` just echoes the name, so printing both
    /// unconditionally produced messages like `[NRT_INVALID]: NRT_INVALID`.
    #[error("Neuron runtime error in {operation}: {detail}")]
    Nrt {
        operation: &'static str,
        status: Status,
        detail: String,
    },

    /// `nrt_init` was called more than once in this process.
    #[error(
        "the Neuron runtime is already initialised in this process; nrt_init must \
         be called exactly once, so share the existing Runtime instead"
    )]
    AlreadyInitialized,

    /// A string handed to the C API contained an interior NUL byte.
    #[error("`{context}` contains an interior NUL byte and cannot be passed to C")]
    InteriorNul { context: &'static str },

    /// A buffer's length does not match the tensor it is being copied into.
    #[error("{name}: tensor holds {tensor_bytes} bytes but {buffer_bytes} were supplied")]
    SizeMismatch {
        name: String,
        tensor_bytes: usize,
        buffer_bytes: usize,
    },

    /// The model has no tensor with the requested name.
    #[error("the model has no {usage} tensor named `{name}`")]
    NoSuchTensor { usage: &'static str, name: String },
}

impl Error {
    /// Builds an [`Error::UnsupportedPlatform`] describing the current target.
    pub(crate) fn unsupported_platform() -> Self {
        Self::UnsupportedPlatform {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        }
    }
}
