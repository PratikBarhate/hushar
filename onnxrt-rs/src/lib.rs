// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! A safe Rust front end over the ONNX Runtime C API.
//!
//! Execution providers select the hardware a model runs on: CoreML on Apple Silicon,
//! CUDA or TensorRT on NVIDIA, ROCm or MIGraphX on AMD. See [`ExecutionProvider`].
//!
//! # Layout
//!
//! * [`sys`] holds raw `bindgen` output generated from `headers/onnxruntime_c_api.h`
//!   (ONNX Runtime v1.28.0, `ORT_API_VERSION` 28), vendored under its MIT licence. The
//!   bindings are checked in, so the crate builds with no libclang and no ONNX Runtime
//!   present. Regenerate with `--features bindgen`.
//! * `libonnxruntime` is resolved at **run time** via `libloading`, never linked at
//!   build time. The whole C API hangs off one exported symbol, `OrtGetApiBase`.
//! * Everything above [`sys`] is safe. Each `Ort*` handle has an owner whose `Drop`
//!   calls the matching `Release*`.
//!
//! # Features and runtime support
//!
//! An execution provider needs both the Cargo feature that compiles its support into
//! this crate ([`enabled_features`] lists the ones enabled) and a `libonnxruntime` that
//! was itself built with that provider. The two failures are reported separately, as
//! [`Error::ProviderNotEnabled`] and [`Error::ProviderUnavailable`].
//!
//! # Finding the runtime at run time
//!
//! Set `ORT_DYLIB_PATH` to an explicit shared library, or install ONNX Runtime
//! somewhere the dynamic loader already searches. See [`Api::load`].
//!
//! # Example
//!
//! ```no_run
//! use onnxrt_rs::{Environment, SessionBuilder};
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create one environment per process and share it.
//! let env = Arc::new(Environment::new("hushar")?);
//!
//! let model_bytes = std::fs::read("model.onnx")?;
//! let session = Arc::new(
//!     SessionBuilder::new(&env)?
//!         .intra_op_threads(1)?
//!         .build_from_memory(&model_bytes)?,
//! );
//!
//! // A batch of 2 rows with 3 features each.
//! let mut features = vec![1.0f32, 2.0, 3.0, 0.5, 1.0, 1.5];
//! let output = session.run_single(&mut features, &[2, 3])?;
//!
//! println!("shape {:?} -> {:?}", output.shape()?, output.as_slice::<f32>()?);
//! # Ok(())
//! # }
//! ```
//!
//! # Thread safety
//!
//! [`Session`] is `Send + Sync`, so an `Arc<Session>` can serve concurrent requests:
//! ONNX Runtime documents `Run` on a shared session as needing no external
//! synchronisation. [`Environment`] is likewise shareable and should be created once
//! per process.

pub mod sys;

mod api;
mod env;
mod error;
mod provider;
mod session;
mod tensor;

pub use api::{Api, DYLIB_PATH_ENV};
pub use env::{Environment, LogLevel};
pub use error::{Error, ErrorCode, Result};
pub use provider::{CoreMlComputeUnits, CoreMlModelFormat, ExecutionProvider, enabled_features};
pub use session::{ExecutionMode, GraphOptimizationLevel, Session, SessionBuilder, TensorSpec};
pub use tensor::{DataType, Element, InputValue, OwnedTensor, TensorView};

/// The `ORT_API_VERSION` these bindings were generated against.
///
/// A loaded runtime older than this cannot supply a compatible API table and
/// [`Api::load`] will fail with [`Error::UnsupportedApiVersion`].
///
/// This is deliberately the *oldest* release the crate needs rather than the newest one
/// available. `OrtApi` is append-only, so a newer runtime still answers `GetApi(28)` with
/// a table whose first 28 versions' worth of members sit at the same offsets, while a
/// runtime older than the requested version returns null and is refused outright.
/// Binding low therefore widens the set of hosts this crate runs on at no cost, and it is
/// what lets one build load ONNX Runtime 1.28 and 1.29 alike.
pub const REQUIRED_API_VERSION: u32 = sys::ORT_API_VERSION;

/// The ONNX Runtime release the checked-in bindings were generated from.
///
/// This is the *compile-time* pairing, and it is also recorded as this crate's SemVer
/// build metadata (`0.1.0+onnxruntime.1.28.0`). The version actually loaded is reported
/// by [`Api::version`] and may legitimately differ: the runtime is resolved at run time,
/// and any build offering [`REQUIRED_API_VERSION`] or newer will work.
pub const BINDINGS_RUNTIME_VERSION: &str = "1.28.0";

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts the crate version's build metadata names [`BINDINGS_RUNTIME_VERSION`].
    #[test]
    fn crate_version_metadata_names_the_bindings_release() {
        let version = env!("CARGO_PKG_VERSION");
        let expected = format!("+onnxruntime.{BINDINGS_RUNTIME_VERSION}");
        assert!(
            version.ends_with(&expected),
            "crate version {version:?} should end with {expected:?}; \
             update Cargo.toml and BINDINGS_RUNTIME_VERSION together"
        );
    }

    /// Asserts the bindings target the `ORT_API_VERSION` of the vendored header.
    #[test]
    fn bindings_target_the_vendored_header_version() {
        assert_eq!(
            REQUIRED_API_VERSION, 28,
            "bindings should target ORT_API_VERSION 28 (ONNX Runtime 1.28.x)"
        );
    }

    /// Asserts each [`Element`] carries the type tag the ONNX standard fixes for it.
    #[test]
    fn element_type_tags_match_the_onnx_spec() {
        assert_eq!(<f32 as Element>::ELEMENT_TYPE, 1);
        assert_eq!(<u8 as Element>::ELEMENT_TYPE, 2);
        assert_eq!(<i8 as Element>::ELEMENT_TYPE, 3);
        assert_eq!(<u16 as Element>::ELEMENT_TYPE, 4);
        assert_eq!(<i16 as Element>::ELEMENT_TYPE, 5);
        assert_eq!(<i32 as Element>::ELEMENT_TYPE, 6);
        assert_eq!(<i64 as Element>::ELEMENT_TYPE, 7);
        assert_eq!(<half::f16 as Element>::ELEMENT_TYPE, 10);
        assert_eq!(<f64 as Element>::ELEMENT_TYPE, 11);
        assert_eq!(<u32 as Element>::ELEMENT_TYPE, 12);
        assert_eq!(<u64 as Element>::ELEMENT_TYPE, 13);
        assert_eq!(<half::bf16 as Element>::ELEMENT_TYPE, 16);
    }

    /// Asserts each [`Element`] is, in Rust and in [`DataType`], the width ONNX
    /// declares for it — the mechanically checkable half of the contract that a
    /// Rust type's layout *is* its ONNX element's layout.
    #[test]
    fn element_types_are_the_width_onnx_says_they_are() {
        fn check<T: Element>(expected: usize) {
            let dt = DataType::from_raw(T::ELEMENT_TYPE)
                .unwrap_or_else(|| panic!("{} should be a modelled DataType", T::NAME));
            assert_eq!(
                dt.size_bytes(),
                Some(expected),
                "{} declares {expected} bytes",
                T::NAME
            );
            assert_eq!(
                std::mem::size_of::<T>(),
                expected,
                "{} is {} bytes in Rust but ONNX says {expected}",
                T::NAME,
                std::mem::size_of::<T>()
            );
        }
        check::<i8>(1);
        check::<u8>(1);
        check::<i16>(2);
        check::<u16>(2);
        check::<half::f16>(2);
        check::<half::bf16>(2);
        check::<i32>(4);
        check::<u32>(4);
        check::<f32>(4);
        check::<i64>(8);
        check::<u64>(8);
        check::<f64>(8);
    }

    /// Asserts every [`DataType`] survives a round trip through its ONNX tag, and that
    /// no two share a tag.
    #[test]
    fn data_type_round_trips_through_its_onnx_tag() {
        let all = [
            DataType::Bool,
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::F16,
            DataType::Bf16,
            DataType::F32,
            DataType::F64,
            DataType::String,
        ];
        for dt in all {
            assert_eq!(
                DataType::from_raw(dt.to_raw()),
                Some(dt),
                "{dt} should survive a round trip through its ONNX tag"
            );
        }
        let mut tags: Vec<u32> = all.iter().map(|d| d.to_raw()).collect();
        tags.sort_unstable();
        let before = tags.len();
        tags.dedup();
        assert_eq!(before, tags.len(), "two DataTypes share an ONNX tag");
    }

    /// Asserts an ONNX element type this crate does not model maps to `None`, rather
    /// than to something of the same width, and that strings report no fixed size.
    #[test]
    fn unmodelled_element_types_are_reported_not_guessed() {
        let complex64 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
        assert_eq!(DataType::from_raw(complex64), None);
        assert_eq!(
            DataType::String.size_bytes(),
            None,
            "strings are variable length"
        );
    }

    /// Asserts a leading dynamic axis resolves to the row count and `row_width` is the
    /// product of the trailing axes, while a dynamic axis that is *not* the batch is
    /// left as `-1` and yields no row width.
    #[test]
    fn tensor_spec_resolves_a_dynamic_batch_axis() {
        let spec = TensorSpec {
            name: "numbers".to_owned(),
            data_type: DataType::F32,
            shape: vec![-1, 3],
        };
        assert_eq!(spec.concrete_shape(8), vec![8, 3]);
        assert_eq!(spec.row_width(), Some(3));

        let forecast = TensorSpec {
            name: "horizon".to_owned(),
            data_type: DataType::F32,
            shape: vec![-1, 24, 2],
        };
        assert_eq!(forecast.concrete_shape(4), vec![4, 24, 2]);
        assert_eq!(forecast.row_width(), Some(48));

        let ragged = TensorSpec {
            name: "tokens".to_owned(),
            data_type: DataType::I64,
            shape: vec![-1, -1],
        };
        assert_eq!(ragged.concrete_shape(2), vec![2, -1]);
        assert_eq!(ragged.row_width(), None);
    }

    /// Asserts a nonexistent library is an [`Error::LibraryLoad`] naming the path
    /// tried, not a panic.
    #[test]
    fn missing_library_is_a_clean_error_not_a_panic() {
        let err = Api::load_from("/nonexistent/libonnxruntime.dylib")
            .expect_err("loading a nonexistent library must fail");
        assert!(
            matches!(err, Error::LibraryLoad { .. }),
            "expected a LibraryLoad error, got {err:?}"
        );
        assert!(
            err.to_string()
                .contains("/nonexistent/libonnxruntime.dylib")
        );
    }

    /// Asserts a failed load reports the dynamic loader's own reason, not just
    /// `libloading`'s bare "dlopen failed" — the reason sits one level down in
    /// `source()`. The appended text is platform-specific, so only its presence is
    /// asserted.
    #[test]
    fn library_load_error_reports_the_underlying_reason() {
        let err = Api::load_from("/nonexistent/libonnxruntime.dylib")
            .expect_err("loading a nonexistent library must fail");
        let message = err.to_string();
        assert!(
            message.len() > "dlopen failed".len() + 40,
            "error message looks truncated to libloading's bare Display: {message}"
        );
        let bare = "dlopen failed";
        let tail = message
            .split_once(bare)
            .map(|(_, rest)| rest.trim_start_matches([':', ' ']))
            .unwrap_or("");
        assert!(
            !tail.is_empty(),
            "no underlying loader reason was appended after {bare:?}: {message}"
        );
    }
}
