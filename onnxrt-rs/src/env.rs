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
/// ONNX Runtime expects one environment per process, created before any session and
/// outliving all of them. Wrap it in an `Arc` and hand clones to whatever needs to build
/// sessions.
///
/// [`Send`] and [`Sync`] rest on `OrtEnv` being a shareable, process-wide object: ONNX
/// Runtime serialises access to the logger and thread pools internally, and this wrapper
/// hands out only `&Environment`, never `&mut`.
#[derive(Debug)]
pub struct Environment {
    pub(crate) api: Api,
    pub(crate) ptr: *mut sys::OrtEnv,
}

unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl Environment {
    /// The process-wide environment, created on first use.
    ///
    /// **Prefer this over [`Self::new`].** ONNX Runtime allows only one environment using
    /// the default logger to exist at a time; creating a second fails with
    ///
    /// ```text
    /// Only one instance of LoggingManager created with
    /// InstanceType::Default can exist at any point in time.
    /// ```
    ///
    /// This function serialises creation and hands out clones of a single environment, so
    /// concurrent first use is safe: two threads loading a model at once, or two tests
    /// running in parallel. A poisoned lock is recovered from rather than propagated, so a
    /// panicking caller does not break later ones.
    ///
    /// The environment is cached for the lifetime of the process and is never released, as
    /// ONNX Runtime requires: it must outlive every session created in it.
    ///
    /// `log_id` applies only if this call is the one that creates the environment; later
    /// calls receive the existing one and ignore it.
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

        let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(existing) = guard.as_ref() {
            return Ok(Arc::clone(existing));
        }

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
    /// Telemetry is switched off; see [`Self::enable_telemetry`].
    ///
    /// On success the returned value owns the `OrtEnv` and releases it on drop.
    pub fn with_api(api: Api, log_id: &str, level: LogLevel) -> Result<Self> {
        let log_id = CString::new(log_id).map_err(|_| Error::InteriorNul {
            context: "environment log id",
        })?;
        let create_env = ort_fn!(api, CreateEnv);

        let mut ptr: *mut sys::OrtEnv = ptr::null_mut();
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
    /// Called for every environment this crate creates, because ONNX Runtime enables
    /// telemetry by default and it can abort the process at exit. Telemetry lives in a
    /// static whose destructor flushes on the way out; with enough session churn that
    /// flush deadlocks on its own mutex and the process dies with `SIGABRT` *after* all
    /// work has completed successfully:
    ///
    /// ```text
    /// onnxruntime::PosixTelemetry::~PosixTelemetry()
    ///   -> Microsoft::Applications::Events::LogManagerImpl::Flush()
    ///     -> std::mutex::lock()  ->  __psynch_mutexwait
    /// ```
    ///
    /// A non-zero exit on shutdown is not cosmetic for a service: orchestrators read it as
    /// a failed task.
    fn disable_telemetry(&self) -> Result<()> {
        let disable = ort_fn!(self.api, DisableTelemetryEvents);
        unsafe { self.api.check(disable(self.ptr)) }
    }

    /// Re-enables ONNX Runtime's telemetry events.
    ///
    /// Off by default because it has crashed the process at exit: ORT flushes telemetry
    /// from a static destructor, and with enough session churn that flush deadlocks on its
    /// own mutex and aborts *after* all work has succeeded.
    ///
    /// Call this only if you need the telemetry and accept the exit risk.
    pub fn enable_telemetry(&self) -> Result<()> {
        let enable = ort_fn!(self.api, EnableTelemetryEvents);
        unsafe { self.api.check(enable(self.ptr)) }
    }

    /// The API table this environment was created with.
    pub fn api(&self) -> Api {
        self.api
    }
}

impl Drop for Environment {
    /// Releases the `OrtEnv`.
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseEnv {
            unsafe { release(self.ptr) };
        }
    }
}
