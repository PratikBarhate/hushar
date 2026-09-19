// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Runtime discovery of the ONNX Runtime shared library and its API table.
//!
//! The entire ONNX Runtime C API is reached through a single exported symbol,
//! `OrtGetApiBase`, which hands back a struct of function pointers. Loading is therefore
//! one `dlsym`, and every other call is an indirect jump through that table.

use std::ffi::{CStr, OsStr, OsString};
use std::sync::OnceLock;

use crate::error::{Error, Result};
use crate::sys;

/// Environment variable naming an explicit shared library to load.
pub const DYLIB_PATH_ENV: &str = "ORT_DYLIB_PATH";

static API: OnceLock<Api> = OnceLock::new();

/// A loaded ONNX Runtime API table.
///
/// Cheap to copy: it is a reference to the runtime's static function-pointer
/// table plus the version string captured when the library was opened.
///
/// `Send + Sync` because `OrtApi` is an immutable table of function pointers that ONNX
/// Runtime creates once and never mutates. Whether the functions themselves may be called
/// concurrently is a per-object question, answered on the wrapper types (see
/// [`Session`](crate::Session)).
#[derive(Debug, Clone, Copy)]
pub struct Api {
    inner: &'static sys::OrtApi,
    version: &'static str,
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    /// Loads ONNX Runtime from the default location and caches it for the
    /// lifetime of the process.
    ///
    /// If `ORT_DYLIB_PATH` is set it is used verbatim; otherwise the platform's
    /// usual library names are tried in turn and resolved through the dynamic
    /// loader's normal search path (`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`,
    /// `PATH`, rpath, and so on).
    ///
    /// The first successful load wins; later calls return the same table and do
    /// not re-open the library.
    pub fn load() -> Result<Self> {
        if let Some(api) = API.get() {
            return Ok(*api);
        }
        let api = Self::load_uncached()?;
        Ok(*API.get_or_init(|| api))
    }

    /// Loads ONNX Runtime from an explicit path, bypassing the process cache.
    ///
    /// The library is deliberately never unloaded: ONNX Runtime objects hold pointers
    /// into its code and data, so unmapping it while any
    /// [`Environment`](crate::Environment) or [`Session`](crate::Session) is alive would
    /// be undefined behaviour. Both the library handle and the version string are
    /// leaked, one allocation per load, which is also what keeps [`Api`] `Copy`.
    ///
    /// `OrtGetApiBase` is declared by the vendored header with C linkage, no arguments
    /// and a pointer return. The table it returns has static storage duration, and the
    /// library stays mapped for the rest of the process, so promoting the borrow to
    /// `'static` is correct.
    ///
    /// # Errors
    ///
    /// [`Error::LibraryLoad`] if the library cannot be opened,
    /// [`Error::MissingEntryPoint`] if it does not export `OrtGetApiBase`,
    /// [`Error::MissingApiFunction`] if the API base carries no `GetApi`, and
    /// [`Error::UnsupportedApiVersion`] if the loaded runtime is older than the header
    /// these bindings were generated from; such a runtime returns null rather than a
    /// partially populated table.
    pub fn load_from(path: impl AsRef<OsStr>) -> Result<Self> {
        let path = path.as_ref();
        let library =
            unsafe { libloading::Library::new(path) }.map_err(|source| Error::LibraryLoad {
                path: path.to_string_lossy().into_owned(),
                source,
            })?;
        let library: &'static libloading::Library = Box::leak(Box::new(library));

        let get_api_base = unsafe {
            library
                .get::<unsafe extern "C" fn() -> *const sys::OrtApiBase>(b"OrtGetApiBase\0")
                .map_err(Error::MissingEntryPoint)?
        };

        unsafe {
            let base = get_api_base();
            if base.is_null() {
                return Err(Error::MissingEntryPoint(libloading::Error::DlSymUnknown));
            }
            let base = &*base;

            let version = base
                .GetVersionString
                .map(|f| f())
                .filter(|p| !p.is_null())
                .map(|p| CStr::from_ptr(p).to_string_lossy().into_owned())
                .unwrap_or_else(|| "<unknown>".to_owned());

            let get_api = base
                .GetApi
                .ok_or(Error::MissingApiFunction("OrtApiBase::GetApi"))?;

            let api = get_api(sys::ORT_API_VERSION);
            if api.is_null() {
                return Err(Error::UnsupportedApiVersion {
                    requested: sys::ORT_API_VERSION,
                    library_version: version,
                });
            }

            Ok(Self {
                inner: &*api,
                version: Box::leak(version.into_boxed_str()),
            })
        }
    }

    /// The ONNX Runtime version string reported by the loaded library, for
    /// example `"1.29.0"`.
    pub fn version(&self) -> &'static str {
        self.version
    }

    /// The raw API table.
    ///
    /// Escape hatch for the parts of ONNX Runtime this crate does not wrap yet.
    /// Calls made through it are entirely unchecked.
    pub fn raw(&self) -> &'static sys::OrtApi {
        self.inner
    }

    /// Turns a possibly-null `OrtStatus` into a `Result`, taking ownership of
    /// the status.
    ///
    /// # Safety
    ///
    /// `status` must be null or a status produced by this API table.
    pub(crate) unsafe fn check(&self, status: sys::OrtStatusPtr) -> Result<()> {
        if status.is_null() {
            Ok(())
        } else {
            Err(unsafe { Error::from_status(self.inner, status) })
        }
    }

    fn load_uncached() -> Result<Self> {
        if let Some(explicit) = std::env::var_os(DYLIB_PATH_ENV) {
            return Self::load_from(explicit);
        }

        let mut last_error = None;
        for candidate in default_library_names() {
            match Self::load_from(candidate) {
                Ok(api) => return Ok(api),
                Err(e) => last_error = Some(e),
            }
        }
        Err(last_error.unwrap_or_else(|| Error::LibraryLoad {
            path: "<no candidates for this platform>".to_owned(),
            source: libloading::Error::DlOpenUnknown,
        }))
    }
}

/// The library names tried when `ORT_DYLIB_PATH` is unset.
///
/// The versioned name comes first. `libonnxruntime.so.1` is the SONAME recorded in the
/// library's own `DT_SONAME`, so the dynamic loader can resolve it wherever the runtime
/// is installed, whereas `libonnxruntime.so` is the *linker* name and is conventionally
/// shipped only in a `-dev`/`-devel` package. Preferring the SONAME also selects the
/// major version this crate was built against on a host carrying several side by side.
///
/// macOS follows the same reasoning, its versioned install name being
/// `@rpath/libonnxruntime.1.dylib`. Windows has no SONAME concept and its DLL carries no
/// version in the name.
fn default_library_names() -> Vec<OsString> {
    let names: &[&str] = if cfg!(target_os = "windows") {
        &["onnxruntime.dll"]
    } else if cfg!(target_vendor = "apple") {
        &["libonnxruntime.1.dylib", "libonnxruntime.dylib"]
    } else {
        &["libonnxruntime.so.1", "libonnxruntime.so"]
    };
    names.iter().map(OsString::from).collect()
}

/// Convenience for pulling a function pointer out of the API table with a
/// descriptive error instead of an `unwrap`.
macro_rules! ort_fn {
    ($api:expr, $name:ident) => {
        $api.raw()
            .$name
            .ok_or($crate::error::Error::MissingApiFunction(stringify!($name)))?
    };
}
pub(crate) use ort_fn;
