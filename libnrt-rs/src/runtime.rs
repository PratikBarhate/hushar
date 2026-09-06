// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! The process-wide Neuron runtime handle.

use std::ffi::{CStr, CString, OsStr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};
use crate::symbols::Symbols;
use crate::sys;

/// Which framework is driving the runtime, reported to `nrt_init`.
///
/// This only affects Neuron's telemetry and logging. Use
/// [`Framework::None`] from a plain Rust service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Framework {
    #[default]
    None,
    TensorFlow,
    PyTorch,
    MxNet,
}

impl Framework {
    fn to_raw(self) -> sys::nrt_framework_type_t {
        match self {
            Self::None => sys::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_NO_FW,
            Self::TensorFlow => sys::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_TENSORFLOW,
            Self::PyTorch => sys::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_PYTORCH,
            Self::MxNet => sys::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_MXNET,
        }
    }
}

/// The Neuron runtime library version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub maintenance: u64,
    pub detail: String,
    pub git_hash: String,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}",
            self.major, self.minor, self.patch, self.maintenance
        )
    }
}

/// Guards against a second `nrt_init` in the same process.
///
/// `nrt_init`/`nrt_close` are process-global, so a second init while the first
/// is live would corrupt shared state. This flag turns that into an error.
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// An initialised Neuron runtime.
///
/// Calls `nrt_init` on construction and `nrt_close` on drop. Everything that
/// touches a Neuron device holds an `Arc<Runtime>`, so the runtime cannot be
/// closed while a [`Model`](crate::Model) or [`Tensor`](crate::Tensor) is still
/// alive; the reference counts enforce shutdown ordering for you.
pub struct Runtime {
    pub(crate) symbols: &'static Symbols,
}

// SAFETY: libnrt is documented as supporting concurrent execution from multiple
// threads, and this type exposes only `&self` methods over an immutable symbol
// table. The one piece of shared mutable state, the `nrt_init`/`nrt_close`
// lifecycle, is serialised by the `INITIALIZED` flag.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    /// Initialises the Neuron runtime, loading `libnrt` from the default
    /// location.
    ///
    /// Fails with [`Error::UnsupportedPlatform`] off Linux and
    /// [`Error::LibraryLoad`] when the SDK is not installed, so a caller can
    /// fall back to a CPU backend instead of aborting.
    pub fn init() -> Result<Arc<Self>> {
        Self::init_with(Framework::None, None, None)
    }

    /// Initialises the runtime with an explicit framework tag and library path.
    ///
    /// `library` overrides both `NRT_DYLIB_PATH` and the default soname.
    pub fn init_with(
        framework: Framework,
        framework_version: Option<&str>,
        library: Option<&OsStr>,
    ) -> Result<Arc<Self>> {
        let symbols = Symbols::load(library)?;

        // Claim the process-wide slot before calling into C, so two threads
        // racing here cannot both reach `nrt_init`.
        if INITIALIZED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Error::AlreadyInitialized);
        }

        let fw_version = match framework_version {
            Some(v) => Some(CString::new(v).map_err(|_| Error::InteriorNul {
                context: "framework version",
            })?),
            None => None,
        };
        let fw_version_ptr = fw_version.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        // SAFETY: `fw_version_ptr` is either null (documented as acceptable) or
        // a valid NUL-terminated string that outlives the call. The final
        // argument is the FAL version, which has no meaning for a non-framework
        // caller, so null is passed.
        let status =
            unsafe { (symbols.nrt_init)(framework.to_raw(), fw_version_ptr, std::ptr::null()) };

        if let Err(e) = symbols.check("nrt_init", status) {
            // Release the slot so a caller can retry after fixing the cause.
            INITIALIZED.store(false, Ordering::Release);
            return Err(e);
        }

        Ok(Arc::new(Self { symbols }))
    }

    /// The `libnrt` version.
    pub fn version(&self) -> Result<Version> {
        let mut raw = sys::nrt_version_t {
            rt_major: 0,
            rt_minor: 0,
            rt_patch: 0,
            rt_maintenance: 0,
            rt_detail: [0; 128],
            git_hash: [0; 64],
        };
        // SAFETY: `raw` is a valid, fully initialised struct and we pass its
        // exact size, which is how libnrt decides how much to fill in.
        unsafe {
            let status = (self.symbols.nrt_get_version)(&mut raw, std::mem::size_of_val(&raw));
            self.symbols.check("nrt_get_version", status)?;
        }

        Ok(Version {
            major: raw.rt_major,
            minor: raw.rt_minor,
            patch: raw.rt_patch,
            maintenance: raw.rt_maintenance,
            detail: fixed_array_to_string(&raw.rt_detail),
            git_hash: fixed_array_to_string(&raw.git_hash),
        })
    }

    /// Total virtual NeuronCores on this instance.
    pub fn total_neuron_cores(&self) -> Result<u32> {
        let mut count = 0u32;
        // SAFETY: valid out-parameter.
        unsafe {
            let status = (self.symbols.nrt_get_total_vnc_count)(&mut count);
            self.symbols.check("nrt_get_total_vnc_count", status)?;
        }
        Ok(count)
    }

    /// Virtual NeuronCores visible to this process, which `NEURON_RT_VISIBLE_CORES`
    /// can restrict.
    pub fn visible_neuron_cores(&self) -> Result<u32> {
        let mut count = 0u32;
        // SAFETY: valid out-parameter.
        unsafe {
            let status = (self.symbols.nrt_get_visible_vnc_count)(&mut count);
            self.symbols.check("nrt_get_visible_vnc_count", status)?;
        }
        Ok(count)
    }
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: `nrt_init` succeeded for this instance, and `Arc` ordering
        // guarantees every model and tensor derived from it is already dropped.
        unsafe { (self.symbols.nrt_close)() };
        INITIALIZED.store(false, Ordering::Release);
    }
}

/// Reads a NUL-padded fixed-size C char array into a `String`.
fn fixed_array_to_string(buf: &[std::os::raw::c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Reads a NUL-terminated C string out of a fixed-size array, borrowing where
/// possible.
pub(crate) fn cstr_from_array(buf: &[std::os::raw::c_char]) -> String {
    // SAFETY: `buf` comes from a C struct field that libnrt NUL-terminates. The
    // explicit bound keeps `from_bytes_until_nul` from running off the end even
    // if it did not.
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len()) };
    CStr::from_bytes_until_nul(bytes)
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}
