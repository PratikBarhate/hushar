// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! The ONNX Runtime environment: process-wide logging and thread pools.

use std::ffi::CString;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use crate::api::{Api, ort_fn};
use crate::error::{Error, Result};
use crate::sys;

/// How much ONNX Runtime should log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    Verbose,
    Info,
    #[default]
    Warning,
    Error,
    Fatal,
}

impl LogLevel {
    fn to_raw(self) -> sys::OrtLoggingLevel {
        match self {
            Self::Verbose => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
            Self::Info => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
            Self::Warning => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
            Self::Error => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
            Self::Fatal => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL,
        }
    }
}

/// An `OrtEnv`, which owns the runtime's logger and shared thread pools.
///
/// ONNX Runtime expects one environment per process, created before any session
/// and outliving all of them. Wrap it in an `Arc` and hand clones to whatever
/// needs to build sessions.
#[derive(Debug)]
pub struct Environment {
    pub(crate) api: Api,
    pub(crate) ptr: *mut sys::OrtEnv,
}

// SAFETY: `OrtEnv` is documented as a shareable, process-wide object; ONNX
// Runtime serialises access to the logger and thread pools it owns internally.
// We hand out only `&Environment` (never `&mut`), so no wrapper-level mutation
// can race either.
unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl Environment {
    /// The process-wide environment, created on first use.
    ///
    /// **Prefer this over [`Self::new`].** ONNX Runtime allows only one
    /// environment using the default logger to exist at a time; creating a second
    /// fails with
    ///
    /// ```text
    /// Only one instance of LoggingManager created with
    /// InstanceType::Default can exist at any point in time.
    /// ```
    ///
    /// That makes [`Self::new`] a race whenever two threads might load a model at
    /// once — reloading a model while serving, or simply two tests running in
    /// parallel. This function serialises creation and hands out clones of a
    /// single environment, so those cases are safe.
    ///
    /// The environment is cached for the lifetime of the process and is never
    /// released, which is what ONNX Runtime wants: it must outlive every session
    /// created in it.
    ///
    /// `log_id` applies only if this call is the one that creates the
    /// environment; later calls receive the existing one and ignore it.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let env = onnxrt_rs::Environment::shared("my-service")?;
    /// let also_env = onnxrt_rs::Environment::shared("ignored")?;
    /// assert!(std::sync::Arc::ptr_eq(&env, &also_env));
    /// # Ok(())
    /// # }
    /// ```
    pub fn shared(log_id: &str) -> Result<Arc<Self>> {
        static SHARED: OnceLock<Mutex<Option<Arc<Environment>>>> = OnceLock::new();
        let slot = SHARED.get_or_init(|| Mutex::new(None));

        // Recover from a poisoned lock rather than propagating it: the guarded
        // value is an `Option<Arc<_>>` that a panicking creator cannot have left
        // in an inconsistent state.
        let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(existing) = guard.as_ref() {
            return Ok(Arc::clone(existing));
        }

        // Held across creation deliberately: this is the serialisation that makes
        // concurrent first-use safe.
        let env = Arc::new(Self::new(log_id)?);
        *guard = Some(Arc::clone(&env));
        Ok(env)
    }

    /// Creates a *new* environment, loading ONNX Runtime from the default
    /// location.
    ///
    /// Prefer [`Self::shared`]. ONNX Runtime permits only one default-logger
    /// environment per process, so calling this while another environment is
    /// alive fails. Use it only when you genuinely need a separate environment
    /// and can guarantee the previous one has been dropped.
    pub fn new(log_id: &str) -> Result<Self> {
        Self::with_api(Api::load()?, log_id, LogLevel::default())
    }

    /// Creates an environment against an already-loaded [`Api`].
    ///
    /// Telemetry is switched off; see [`Self::enable_telemetry`] for why, and for
    /// how to turn it back on.
    pub fn with_api(api: Api, log_id: &str, level: LogLevel) -> Result<Self> {
        let log_id = CString::new(log_id).map_err(|_| Error::InteriorNul {
            context: "environment log id",
        })?;
        let create_env = ort_fn!(api, CreateEnv);

        let mut ptr: *mut sys::OrtEnv = ptr::null_mut();
        // SAFETY: `log_id` is a valid NUL-terminated string that outlives the
        // call, and `ptr` is a valid out-parameter. On success ONNX Runtime
        // writes an owned `OrtEnv` there, released in `Drop`.
        unsafe {
            api.check(create_env(level.to_raw(), log_id.as_ptr(), &mut ptr))?;
        }
        debug_assert!(!ptr.is_null(), "CreateEnv returned OK with a null env");

        let environment = Self { api, ptr };
        environment.disable_telemetry()?;
        Ok(environment)
    }

    /// Turns off ONNX Runtime's telemetry events.
    ///
    /// Called for every environment this crate creates, because ONNX Runtime
    /// enables telemetry by default and there are two reasons not to want it:
    ///
    /// **It crashes at process exit.** ORT's telemetry lives in a static whose
    /// destructor flushes on the way out. With enough session churn that flush
    /// deadlocks on its own mutex and the process dies with `SIGABRT` *after* all
    /// work has completed successfully:
    ///
    /// ```text
    /// onnxruntime::PosixTelemetry::~PosixTelemetry()
    ///   -> Microsoft::Applications::Events::LogManagerImpl::Flush()
    ///     -> std::mutex::lock()  ->  __psynch_mutexwait
    /// ```
    ///
    /// A non-zero exit on shutdown is not cosmetic for a service: orchestrators
    /// read it as a failed task and may count it toward a restart budget.
    ///
    /// **A service should not phone home.** Inference is on the request path, and
    /// emitting vendor telemetry from it is neither free nor something a caller
    /// consented to.
    fn disable_telemetry(&self) -> Result<()> {
        let disable = ort_fn!(self.api, DisableTelemetryEvents);
        // SAFETY: `self.ptr` is a live env produced by `CreateEnv`.
        unsafe { self.api.check(disable(self.ptr)) }
    }

    /// Re-enables ONNX Runtime's telemetry events.
    ///
    /// Off by default for two reasons. It has crashed the process at exit: ORT
    /// flushes telemetry from a static destructor, and with enough session churn
    /// that flush deadlocks on its own mutex and aborts *after* all work has
    /// succeeded. And a service on a request path has no business phoning home.
    ///
    /// Call this only if you need the telemetry and accept the exit risk.
    pub fn enable_telemetry(&self) -> Result<()> {
        let enable = ort_fn!(self.api, EnableTelemetryEvents);
        // SAFETY: `self.ptr` is a live env produced by `CreateEnv`.
        unsafe { self.api.check(enable(self.ptr)) }
    }

    /// The API table this environment was created with.
    pub fn api(&self) -> Api {
        self.api
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseEnv {
            // SAFETY: `ptr` was produced by `CreateEnv`, has not been released,
            // and is not used again: `self` is being dropped.
            unsafe { release(self.ptr) };
        }
    }
}
