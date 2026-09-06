// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! A safe Rust front end over the AWS Neuron runtime (`libnrt`) C API.
//!
//! Neuron is the runtime behind Inferentia and Trainium instances. Models are
//! compiled ahead of time by the Neuron compiler into a **NEFF**, which this
//! crate loads and executes; it does not compile ONNX itself. A typical pipeline
//! is `ONNX -> neuronx-cc -> .neff -> this crate`.
//!
//! # How this crate is put together
//!
//! * [`sys`] holds raw `bindgen` output generated from the Neuron SDK headers
//!   (`aws-neuronx-runtime-lib` 2.34.10.0). The bindings are checked in, so the
//!   crate builds on any host; regenerate with `--features bindgen` on a machine
//!   that has the SDK.
//! * The SDK headers themselves are **not** vendored. Unlike ONNX Runtime's MIT
//!   licence, they ship as "All Rights Reserved", so `headers/wrapper.h` includes
//!   them from `NEURON_INCLUDE_DIR` (default `/opt/aws/neuron/include`) instead.
//! * `libnrt.so.1` is opened at **run time** with `libloading`, never linked.
//!   That is what lets one binary run on both Neuron and non-Neuron hosts, and
//!   what lets this crate be a member of a workspace that builds on macOS.
//! * Everything above [`sys`] is safe, with `Drop` on every handle and a written
//!   justification on every `unsafe` block.
//!
//! # Platform support
//!
//! The Neuron runtime is Linux-only and needs a Neuron device. On any other
//! platform [`Runtime::init`] returns [`Error::UnsupportedPlatform`] rather than
//! failing to compile, so a service can carry a Neuron backend and a CPU backend
//! in one binary and choose at startup.
//!
//! # Shutdown ordering
//!
//! `nrt_init` and `nrt_close` are process-global, and closing the runtime while
//! a model is loaded is undefined behaviour. [`Runtime`] is therefore handed out
//! as an `Arc`, and [`Model`], [`Tensor`], and [`TensorSet`] each hold a clone.
//! Reference counting makes the correct teardown order the only one possible, so
//! there is no ordering rule to remember.
//!
//! # Example
//!
//! ```no_run
//! use libnrt_rs::{Model, Runtime};
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialise once per process.
//! let runtime = Runtime::init()?;
//! println!(
//!     "libnrt {} with {} visible NeuronCores",
//!     runtime.version()?,
//!     runtime.visible_neuron_cores()?
//! );
//!
//! // Load a NEFF produced by the Neuron compiler.
//! let neff = std::fs::read("model.neff")?;
//! let model = Model::load(Arc::clone(&runtime), &neff, 0, 1)?;
//!
//! for tensor in model.tensors() {
//!     println!("{:?} {} {:?} {:?}", tensor.usage, tensor.name, tensor.dtype, tensor.shape);
//! }
//!
//! // Buffers sized straight from the NEFF.
//! let (mut inputs, mut outputs) = model.allocate_io(0)?;
//!
//! let input_name = model.inputs().next().expect("model has an input").name.clone();
//! inputs
//!     .get_mut(&input_name)
//!     .expect("just allocated")
//!     .write_f32(&[1.0, 2.0, 3.0])?;
//!
//! model.execute(&inputs, &mut outputs)?;
//!
//! for tensor in outputs.tensors() {
//!     println!("{} -> {:?}", tensor.name(), tensor.read_f32()?);
//! }
//! # Ok(())
//! # }
//! ```

pub mod sys;

mod error;
mod model;
mod runtime;
mod symbols;
mod tensor;

pub use error::{Error, Result, Status};
pub use model::{Model, TensorInfo, Usage};
pub use runtime::{Framework, Runtime, Version};
pub use symbols::DYLIB_PATH_ENV;
pub use tensor::{DataType, Placement, Tensor, TensorSet};

/// Whether this build can talk to a Neuron device at all.
///
/// False on every non-Linux target. A true result means only that the platform
/// is supported, not that `libnrt` is installed or a device is present; call
/// [`Runtime::init`] to find that out.
pub const fn is_supported_platform() -> bool {
    cfg!(target_os = "linux")
}

/// The Neuron SDK release the checked-in bindings were generated from.
///
/// This is the *compile-time* pairing, and it is also recorded as this crate's
/// SemVer build metadata (`0.1.0+libnrt.2.34.10.0`). The version actually loaded
/// is reported by [`Runtime::version`] and may legitimately differ: `libnrt` is
/// resolved at run time via its SONAME `libnrt.so.1`, which stays stable across
/// SDK releases, so any ABI-1 build will load. Compare the two when a bug report
/// needs to say exactly what was running.
pub const BINDINGS_SDK_VERSION: &str = "2.34.10.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_version_metadata_names_the_bindings_release() {
        // Keeps Cargo.toml and BINDINGS_SDK_VERSION from drifting apart.
        let version = env!("CARGO_PKG_VERSION");
        let expected = format!("+libnrt.{BINDINGS_SDK_VERSION}");
        assert!(
            version.ends_with(&expected),
            "crate version {version:?} should end with {expected:?}; \
             update Cargo.toml and BINDINGS_SDK_VERSION together"
        );
    }

    #[test]
    fn status_mapping_round_trips_through_raw_values() {
        // Values are fixed by the SDK's nrt_status.h, so pinning them here
        // catches a mis-generated or hand-edited `sys` module.
        for (status, raw) in [
            (Status::Failure, 1u32),
            (Status::Invalid, 2),
            (Status::InvalidHandle, 3),
            (Status::Resource, 4),
            (Status::Timeout, 5),
            (Status::HardwareError, 6),
            (Status::QueueFull, 7),
            (Status::NotEnoughNeuronCores, 9),
            (Status::UnsupportedNeffVersion, 10),
            (Status::Uninitialized, 13),
            (Status::Closed, 14),
            (Status::ExecBadInput, 1002),
            (Status::ExecCompletedWithNumericalError, 1003),
            (Status::ExecCompletedWithError, 1004),
            (Status::ExecNeuronCoreBusy, 1005),
            (Status::ExecOutOfBounds, 1006),
        ] {
            assert_eq!(status.as_raw(), raw, "{status} should map to {raw}");
            assert_eq!(
                Status::from_raw(raw),
                Some(status),
                "raw {raw} should map back to {status}"
            );
        }
    }

    #[test]
    fn success_is_not_an_error() {
        assert_eq!(Status::from_raw(sys::NRT_STATUS_NRT_SUCCESS), None);
    }

    #[test]
    fn unknown_status_codes_are_preserved() {
        // Forward compatibility: a status this crate has never heard of must not
        // be silently flattened.
        let unknown = 99_999;
        assert_eq!(Status::from_raw(unknown), Some(Status::Other(unknown)));
        assert_eq!(Status::Other(unknown).as_raw(), unknown);
    }

    #[test]
    fn dtype_values_match_the_sdk_header() {
        // NRT_DTYPE_* values are non-contiguous hex in the header, which makes a
        // careless `as` cast wrong; these assertions pin the real mapping.
        assert_eq!(DataType::from_raw(0xA), DataType::Float32);
        assert_eq!(DataType::from_raw(0x7), DataType::Float16);
        assert_eq!(DataType::from_raw(0x6), DataType::BFloat16);
        assert_eq!(DataType::from_raw(0x8), DataType::Int32);
        assert_eq!(DataType::from_raw(0xC), DataType::Int64);
        assert_eq!(DataType::from_raw(0x1), DataType::Uint64);
        assert_eq!(DataType::from_raw(0x2), DataType::Int8);
        assert_eq!(DataType::from_raw(0x3), DataType::Uint8);
    }

    #[test]
    fn dtype_element_sizes_are_consistent() {
        assert_eq!(DataType::Float32.size_of(), Some(4));
        assert_eq!(DataType::BFloat16.size_of(), Some(2));
        assert_eq!(DataType::Int64.size_of(), Some(8));
        assert_eq!(DataType::Fp8E4.size_of(), Some(1));
        assert_eq!(DataType::Unknown(0x42).size_of(), None);
    }

    #[test]
    fn tensor_set_is_an_opaque_void_pointer() {
        // nrt.h declares `typedef void nrt_tensor_set_t`, so the binding must be
        // c_void rather than a generated struct.
        assert_eq!(
            std::mem::size_of::<*mut sys::nrt_tensor_set_t>(),
            std::mem::size_of::<*mut std::os::raw::c_void>()
        );
    }

    #[test]
    fn init_fails_cleanly_on_unsupported_platforms() {
        if is_supported_platform() {
            // On Linux the outcome depends on whether libnrt and a device are
            // present, so there is nothing deterministic to assert here.
            return;
        }
        let err = Runtime::init().expect_err("Neuron must not initialise off Linux");
        assert!(
            matches!(err, Error::UnsupportedPlatform { .. }),
            "expected UnsupportedPlatform, got {err:?}"
        );
        // The message should say which platform was built, to save a support round trip.
        assert!(err.to_string().contains(std::env::consts::OS));
    }
}
