// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Run-time resolution of the `libnrt` symbols this crate uses.
//!
//! `libnrt` exports plain C functions rather than a function-pointer table, so each one
//! is resolved individually with `libloading`. The signatures below are transcribed from
//! the `extern "C"` declarations `bindgen` produced in [`crate::sys`], and must be kept
//! in step with a regenerated `sys` module.
//!
//! Loading at run time rather than linking at build time means one binary is identical
//! on Neuron and non-Neuron hosts, and only fails when something asks for a Neuron
//! device.

use std::ffi::{OsStr, OsString};
use std::os::raw::{c_char, c_int, c_void};

use crate::error::{Error, Result};
use crate::sys;

/// Environment variable naming an explicit `libnrt` to load.
pub const DYLIB_PATH_ENV: &str = "NRT_DYLIB_PATH";

/// Soname the Neuron SDK installs. `libnrt.so` is a symlink to this, and only
/// exists when the `-dev` files are present, so prefer the versioned name.
const DEFAULT_LIBRARY_NAMES: &[&str] = &["libnrt.so.1", "libnrt.so"];

/// The `libnrt` entry points this crate calls.
///
/// Every field is a raw function pointer resolved once at load time. The
/// [`Library`](libloading::Library) they came from is leaked, so the pointers
/// stay valid for the life of the process.
#[allow(non_snake_case)]
pub(crate) struct Symbols {
    pub nrt_init: unsafe extern "C" fn(
        sys::nrt_framework_type_t,
        *const c_char,
        *const c_char,
    ) -> sys::NRT_STATUS,
    pub nrt_close: unsafe extern "C" fn(),
    pub nrt_get_status_as_str: unsafe extern "C" fn(sys::NRT_STATUS) -> *const c_char,
    pub nrt_get_version: unsafe extern "C" fn(*mut sys::nrt_version_t, usize) -> sys::NRT_STATUS,
    pub nrt_get_total_vnc_count: unsafe extern "C" fn(*mut u32) -> sys::NRT_STATUS,
    pub nrt_get_visible_vnc_count: unsafe extern "C" fn(*mut u32) -> sys::NRT_STATUS,

    pub nrt_load: unsafe extern "C" fn(
        *const c_void,
        usize,
        i32,
        i32,
        *mut *mut sys::nrt_model_t,
    ) -> sys::NRT_STATUS,
    pub nrt_unload: unsafe extern "C" fn(*mut sys::nrt_model_t) -> sys::NRT_STATUS,
    pub nrt_execute: unsafe extern "C" fn(
        *mut sys::nrt_model_t,
        *const sys::nrt_tensor_set_t,
        *mut sys::nrt_tensor_set_t,
    ) -> sys::NRT_STATUS,

    pub nrt_get_model_tensor_info: unsafe extern "C" fn(
        *mut sys::nrt_model_t,
        *mut *mut sys::nrt_tensor_info_array_t,
    ) -> sys::NRT_STATUS,
    pub nrt_free_model_tensor_info:
        unsafe extern "C" fn(*mut sys::nrt_tensor_info_array_t) -> sys::NRT_STATUS,

    pub nrt_allocate_tensor_set:
        unsafe extern "C" fn(*mut *mut sys::nrt_tensor_set_t) -> sys::NRT_STATUS,
    pub nrt_destroy_tensor_set: unsafe extern "C" fn(*mut *mut sys::nrt_tensor_set_t),
    pub nrt_add_tensor_to_tensor_set: unsafe extern "C" fn(
        *mut sys::nrt_tensor_set_t,
        *const c_char,
        *mut sys::nrt_tensor_t,
    ) -> sys::NRT_STATUS,

    pub nrt_tensor_allocate: unsafe extern "C" fn(
        sys::nrt_tensor_placement_t,
        c_int,
        usize,
        *const c_char,
        *mut *mut sys::nrt_tensor_t,
    ) -> sys::NRT_STATUS,
    pub nrt_tensor_free: unsafe extern "C" fn(*mut *mut sys::nrt_tensor_t),
    pub nrt_tensor_read: unsafe extern "C" fn(
        *const sys::nrt_tensor_t,
        *mut c_void,
        usize,
        usize,
    ) -> sys::NRT_STATUS,
    pub nrt_tensor_write: unsafe extern "C" fn(
        *mut sys::nrt_tensor_t,
        *const c_void,
        usize,
        usize,
    ) -> sys::NRT_STATUS,
    pub nrt_tensor_get_size: unsafe extern "C" fn(*const sys::nrt_tensor_t) -> usize,
}

/// Resolves one symbol from `$library`, transmuting it to the declared signature.
///
/// The signature transmuted to is transcribed from the `extern "C"` declaration bindgen
/// generated for the same symbol in [`crate::sys`], so the ABI matches whatever `libnrt`
/// exports. Lifting the raw pointer out of the borrowed `libloading::Symbol` is sound
/// only because the library is leaked before this runs, so it outlives every pointer
/// resolved from it.
///
/// # Errors
///
/// [`Error::MissingSymbol`] when `libnrt` does not export the name.
macro_rules! symbol {
    ($library:expr, $name:ident) => {{
        let name = concat!(stringify!($name), "\0").as_bytes();
        let symbol: libloading::Symbol<'_, _> =
            unsafe { $library.get(name) }.map_err(|source| Error::MissingSymbol {
                symbol: stringify!($name),
                source,
            })?;
        *symbol
    }};
}

impl Symbols {
    /// Opens `libnrt` and resolves every symbol this crate needs.
    ///
    /// The library opened is `path` if given, else `$NRT_DYLIB_PATH`, else each name in
    /// [`DEFAULT_LIBRARY_NAMES`] in turn. Both the library and the resolved [`Symbols`]
    /// are leaked deliberately: models and tensors hold pointers into the library's code
    /// and data, and `nrt_close` must still be callable at shutdown, so it must never be
    /// unmapped. That is what makes the `'static` return sound.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedPlatform`] off Linux, [`Error::LibraryLoad`] when no candidate
    /// can be opened (the last failure is reported), or [`Error::MissingSymbol`] when a
    /// symbol is absent. A missing symbol fails here rather than at the first call, so an
    /// SDK too old for this crate is reported at startup.
    pub(crate) fn load(path: Option<&OsStr>) -> Result<&'static Self> {
        if !cfg!(target_os = "linux") {
            return Err(Error::unsupported_platform());
        }

        let candidates: Vec<OsString> = match path {
            Some(p) => vec![p.to_os_string()],
            None => match std::env::var_os(DYLIB_PATH_ENV) {
                Some(p) => vec![p],
                None => DEFAULT_LIBRARY_NAMES.iter().map(OsString::from).collect(),
            },
        };

        let mut last_error = None;
        for candidate in &candidates {
            match unsafe { libloading::Library::new(candidate) } {
                Ok(library) => {
                    let library: &'static libloading::Library = Box::leak(Box::new(library));
                    return Ok(Box::leak(Box::new(Self::resolve(library)?)));
                }
                Err(source) => {
                    last_error = Some(Error::LibraryLoad {
                        path: candidate.to_string_lossy().into_owned(),
                        source,
                    })
                }
            }
        }
        Err(last_error.expect("candidate list is never empty"))
    }

    fn resolve(library: &'static libloading::Library) -> Result<Self> {
        Ok(Self {
            nrt_init: symbol!(library, nrt_init),
            nrt_close: symbol!(library, nrt_close),
            nrt_get_status_as_str: symbol!(library, nrt_get_status_as_str),
            nrt_get_version: symbol!(library, nrt_get_version),
            nrt_get_total_vnc_count: symbol!(library, nrt_get_total_vnc_count),
            nrt_get_visible_vnc_count: symbol!(library, nrt_get_visible_vnc_count),
            nrt_load: symbol!(library, nrt_load),
            nrt_unload: symbol!(library, nrt_unload),
            nrt_execute: symbol!(library, nrt_execute),
            nrt_get_model_tensor_info: symbol!(library, nrt_get_model_tensor_info),
            nrt_free_model_tensor_info: symbol!(library, nrt_free_model_tensor_info),
            nrt_allocate_tensor_set: symbol!(library, nrt_allocate_tensor_set),
            nrt_destroy_tensor_set: symbol!(library, nrt_destroy_tensor_set),
            nrt_add_tensor_to_tensor_set: symbol!(library, nrt_add_tensor_to_tensor_set),
            nrt_tensor_allocate: symbol!(library, nrt_tensor_allocate),
            nrt_tensor_free: symbol!(library, nrt_tensor_free),
            nrt_tensor_read: symbol!(library, nrt_tensor_read),
            nrt_tensor_write: symbol!(library, nrt_tensor_write),
            nrt_tensor_get_size: symbol!(library, nrt_tensor_get_size),
        })
    }

    /// Turns a raw status into a `Result`, attaching `libnrt`'s own description.
    ///
    /// `nrt_get_status_as_str` returns the status *name* for most codes, so the
    /// description is only appended when it differs from the name.
    ///
    /// # Errors
    ///
    /// [`Error::Nrt`] for any status other than `NRT_SUCCESS`.
    pub(crate) fn check(&self, operation: &'static str, status: sys::NRT_STATUS) -> Result<()> {
        match crate::error::Status::from_raw(status) {
            None => Ok(()),
            Some(status) => {
                let text = self.status_text(status.as_raw());
                let name = status.to_string();
                let detail = if text == name {
                    name
                } else {
                    format!("{name} ({text})")
                };
                Err(Error::Nrt {
                    operation,
                    status,
                    detail,
                })
            }
        }
    }

    /// `libnrt`'s description of a status, or `<no description>` if it returns null.
    ///
    /// The pointer returned is to a string owned by `libnrt`, never an owned allocation,
    /// so there is nothing to free and the text is copied out rather than retained.
    fn status_text(&self, status: sys::NRT_STATUS) -> String {
        unsafe {
            let p = (self.nrt_get_status_as_str)(status);
            if p.is_null() {
                "<no description>".to_owned()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}

impl std::fmt::Debug for Symbols {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Symbols").finish_non_exhaustive()
    }
}
