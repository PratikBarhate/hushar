// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Sessions: loading a model and running inference on it.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::api::{Api, ort_fn};
use crate::env::Environment;
use crate::error::{Error, Result};
use crate::provider::ExecutionProvider;
use crate::sys;
use crate::tensor::{DataType, Element, InputValue, OwnedTensor, TensorView};

/// Graph optimisations ONNX Runtime should apply when loading a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphOptimizationLevel {
    Disabled,
    Basic,
    Extended,
    Layout,
    #[default]
    All,
}

impl GraphOptimizationLevel {
    fn to_raw(self) -> sys::GraphOptimizationLevel {
        match self {
            Self::Disabled => sys::GraphOptimizationLevel_ORT_DISABLE_ALL,
            Self::Basic => sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
            Self::Extended => sys::GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
            Self::Layout => sys::GraphOptimizationLevel_ORT_ENABLE_LAYOUT,
            Self::All => sys::GraphOptimizationLevel_ORT_ENABLE_ALL,
        }
    }
}

/// Whether the operators in a graph run one after another or concurrently.
///
/// [`Sequential`](Self::Sequential) is ONNX Runtime's default and runs one node at a
/// time, so [`SessionBuilder::inter_op_threads`] has nothing to schedule and is ignored.
/// [`Parallel`](Self::Parallel) is what gives that pool work, and only pays off on a graph
/// with independent branches -- two towers, a shared trunk with several output heads, a
/// wide path beside a deep one. A chain of transformer blocks has none, and neither do
/// attention heads: those are a dimension inside one batched operator, parallelised by the
/// intra-op pool.
///
/// Two costs come with [`Parallel`](Self::Parallel). Memory pattern optimisation is
/// unavailable, since it traces allocations through a known node order. And the pool
/// cannot be pinned -- ONNX Runtime has an affinity setting for the intra-op pool and none
/// for this one -- so its threads go wherever the mask of the thread that built the
/// session allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    #[default]
    Sequential,
    Parallel,
}

impl ExecutionMode {
    fn to_raw(self) -> sys::ExecutionMode {
        match self {
            Self::Sequential => sys::ExecutionMode_ORT_SEQUENTIAL,
            Self::Parallel => sys::ExecutionMode_ORT_PARALLEL,
        }
    }
}

/// Owned `OrtSessionOptions`, configured through [`SessionBuilder`].
struct SessionOptions {
    api: Api,
    ptr: *mut sys::OrtSessionOptions,
}

impl Drop for SessionOptions {
    /// Releases the options object.
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseSessionOptions {
            unsafe { release(self.ptr) };
        }
    }
}

/// Builds a [`Session`].
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use onnxrt_rs::{Environment, SessionBuilder};
/// use std::sync::Arc;
///
/// let env = Arc::new(Environment::new("hushar")?);
/// let model = std::fs::read("model.onnx")?;
/// let session = SessionBuilder::new(&env)?
///     .intra_op_threads(1)?
///     .build_from_memory(&model)?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder<'env> {
    env: &'env Environment,
    options: SessionOptions,
}

impl<'env> SessionBuilder<'env> {
    /// Starts building a session in `env`.
    ///
    /// The builder borrows `env`, which must therefore outlive it. The options object it
    /// creates is released when the builder is dropped.
    pub fn new(env: &'env Environment) -> Result<Self> {
        let api = env.api();
        let create = ort_fn!(api, CreateSessionOptions);
        let mut ptr: *mut sys::OrtSessionOptions = ptr::null_mut();
        unsafe { api.check(create(&mut ptr))? };

        Ok(Self {
            env,
            options: SessionOptions { api, ptr },
        })
    }

    /// Threads used to parallelise work *within* a single operator.
    ///
    /// Set this to 1 when the caller already saturates the machine with concurrent
    /// requests; extra intra-op threads then only add contention.
    pub fn intra_op_threads(self, threads: i32) -> Result<Self> {
        let f = ort_fn!(self.options.api, SetIntraOpNumThreads);
        unsafe { self.options.api.check(f(self.options.ptr, threads))? };
        Ok(self)
    }

    /// Threads used to run independent operators concurrently.
    ///
    /// Has no effect unless [`Self::execution_mode`] is
    /// [`ExecutionMode::Parallel`]: the sequential executor runs one node at a time, so
    /// this pool is built and never given work. The two are set independently here, which
    /// means a count without the mode is accepted and silently does nothing -- ONNX
    /// Runtime's own behaviour, mirrored rather than corrected.
    pub fn inter_op_threads(self, threads: i32) -> Result<Self> {
        let f = ort_fn!(self.options.api, SetInterOpNumThreads);
        unsafe { self.options.api.check(f(self.options.ptr, threads))? };
        Ok(self)
    }

    /// Whether operators run one at a time or concurrently; defaults to
    /// [`ExecutionMode::Sequential`].
    ///
    /// See [`ExecutionMode`] for what each mode costs. This is the switch that makes
    /// [`Self::inter_op_threads`] mean anything.
    pub fn execution_mode(self, mode: ExecutionMode) -> Result<Self> {
        let f = ort_fn!(self.options.api, SetSessionExecutionMode);
        unsafe { self.options.api.check(f(self.options.ptr, mode.to_raw()))? };
        Ok(self)
    }

    /// Sets one of ONNX Runtime's string-keyed session options.
    ///
    /// For settings that have no dedicated setter in the C API and are reached by name
    /// instead. The keys are listed in `onnxruntime_session_options_config_keys.h`; the
    /// ones that matter for serving are
    ///
    /// | Key | Effect |
    /// |---|---|
    /// | `session.intra_op_thread_affinities` | which logical CPUs the intra-op threads may run on |
    /// | `session.intra_op.allow_spinning` | `0` stops idle intra-op threads burning cycles |
    /// | `session.inter_op.allow_spinning` | the same for the inter-op pool |
    ///
    /// # Affinity format
    ///
    /// Semicolon-separated, **one entry per intra-op thread other than the calling
    /// thread**, so `intra_op_threads(n)` wants `n - 1` entries. Each entry is a
    /// comma-separated list of logical CPU ids that thread may use:
    ///
    /// ```text
    ///   "1;2"            two extra threads, pinned to CPU 1 and CPU 2
    ///   "3,4;5,6"        two extra threads, each free to use a pair
    /// ```
    ///
    /// A count that disagrees with `intra_op_threads` is rejected by ONNX Runtime when
    /// the session is built, not here.
    ///
    /// # Errors
    ///
    /// Fails if the key is not one this runtime knows, or the value will not parse.
    pub fn config_entry(self, key: &str, value: &str) -> Result<Self> {
        let f = ort_fn!(self.options.api, AddSessionConfigEntry);
        let key = CString::new(key).map_err(|_| Error::InteriorNul {
            context: "session config key",
        })?;
        let value = CString::new(value).map_err(|_| Error::InteriorNul {
            context: "session config value",
        })?;
        unsafe {
            self.options
                .api
                .check(f(self.options.ptr, key.as_ptr(), value.as_ptr()))?
        };
        Ok(self)
    }

    /// Graph optimisation level; defaults to
    /// [`GraphOptimizationLevel::All`].
    pub fn optimization_level(self, level: GraphOptimizationLevel) -> Result<Self> {
        let f = ort_fn!(self.options.api, SetSessionGraphOptimizationLevel);
        unsafe {
            self.options
                .api
                .check(f(self.options.ptr, level.to_raw()))?
        };
        Ok(self)
    }

    /// Registers an execution provider (a hardware backend).
    ///
    /// Call this more than once to register several: ONNX Runtime assigns each node to the
    /// first registered provider that can run it, so **call order is priority order**. The
    /// CPU provider is always the implicit final fallback, so a model a device cannot fully
    /// handle still runs rather than failing.
    ///
    /// Fails with [`Error::ProviderUnavailable`], naming the providers the loaded library
    /// *does* have, if the requested one was not compiled in. Providers live inside
    /// `libonnxruntime`, so this can only be known at run time; see
    /// [`Api::available_providers`].
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use onnxrt_rs::{Environment, ExecutionProvider, SessionBuilder};
    ///
    /// let env = Environment::new("hushar")?;
    /// let model = std::fs::read("model.onnx")?;
    /// let session = SessionBuilder::new(&env)?
    ///     // Try the Neural Engine, then fall back to CPU for unsupported nodes.
    ///     .execution_provider(&ExecutionProvider::core_ml())?
    ///     .build_from_memory(&model)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn execution_provider(self, provider: &ExecutionProvider) -> Result<Self> {
        provider.append(self.options.api, self.options.ptr)?;
        Ok(self)
    }

    /// Loads a model from an in-memory ONNX protobuf.
    ///
    /// ONNX Runtime copies whatever it needs out of `model` during this call, so the
    /// borrow does not need to outlive it; the environment, however, must outlive the
    /// returned session.
    pub fn build_from_memory(self, model: &[u8]) -> Result<Session> {
        let api = self.options.api;
        let create = ort_fn!(api, CreateSessionFromArray);
        let mut ptr: *mut sys::OrtSession = ptr::null_mut();

        unsafe {
            api.check(create(
                self.env.ptr,
                model.as_ptr().cast::<std::os::raw::c_void>(),
                model.len(),
                self.options.ptr,
                &mut ptr,
            ))?;
        }

        let mut session = Session {
            api,
            ptr,
            input_names: Vec::new(),
            output_names: Vec::new(),
        };
        session.input_names = session.load_io_names(IoKind::Input)?;
        session.output_names = session.load_io_names(IoKind::Output)?;
        Ok(session)
    }
}

#[derive(Debug, Clone, Copy)]
enum IoKind {
    Input,
    Output,
}

impl IoKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }
}

/// One input or output a model declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// The name the model declares. Inputs are matched by this in
    /// [`Session::run_named`].
    pub name: String,
    /// The element type the model expects or produces.
    pub data_type: DataType,
    /// The declared shape. `-1` marks a dynamic axis, which a batch dimension
    /// almost always is.
    pub shape: Vec<i64>,
}

impl TensorSpec {
    /// The shape with dynamic axes resolved against `batch_size`.
    ///
    /// Only a leading dynamic axis is filled in; a dynamic axis anywhere else is a property
    /// of the model that the caller has to supply, and is left as `-1`.
    pub fn concrete_shape(&self, batch_size: usize) -> Vec<i64> {
        let mut shape = self.shape.clone();
        if let Some(first) = shape.first_mut()
            && *first < 0
        {
            *first = batch_size as i64;
        }
        shape
    }

    /// Elements per row, that is, the product of every axis after the first.
    ///
    /// `None` if any of those axes is dynamic.
    pub fn row_width(&self) -> Option<usize> {
        self.shape.iter().skip(1).try_fold(1usize, |acc, &d| {
            if d < 0 { None } else { Some(acc * d as usize) }
        })
    }
}

/// A loaded model, ready to run.
///
/// Cloning is not supported; share it with an `Arc`. Concurrent [`Session::run`] calls on
/// one session are safe and are the intended way to serve traffic. The [`Environment`] the
/// session was built in must outlive it.
///
/// [`Send`] and [`Sync`] rest on ONNX Runtime documenting `Run` as safe to invoke on one
/// session from multiple threads with no external synchronisation (its
/// `docs/NotesOnThreading.md`, and microsoft/onnxruntime#114). This wrapper exposes only
/// `&self` methods, and the cached name vectors are immutable after construction.
#[derive(Debug)]
pub struct Session {
    api: Api,
    ptr: *mut sys::OrtSession,
    input_names: Vec<CString>,
    output_names: Vec<CString>,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    /// The model's input names, in index order.
    pub fn input_names(&self) -> Vec<&str> {
        self.input_names
            .iter()
            .map(|c| c.to_str().unwrap_or_default())
            .collect()
    }

    /// The model's output names, in index order.
    pub fn output_names(&self) -> Vec<&str> {
        self.output_names
            .iter()
            .map(|c| c.to_str().unwrap_or_default())
            .collect()
    }

    /// Runs the model on inputs of mixed element type, matched positionally.
    ///
    /// The general entry point: [`InputValue`] erases the element type, so one call can mix
    /// a float tensor with a string tensor. [`Session::run`] is the convenience form for a
    /// model whose inputs all hold the same [`Element`].
    ///
    /// Inputs must be in the model's declared order; [`Session::run_named`] matches them by
    /// name instead.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use onnxrt_rs::{Environment, InputValue, SessionBuilder};
    ///
    /// let env = Environment::new("hushar")?;
    /// let session = SessionBuilder::new(&env)?.build_from_memory(&std::fs::read("m.onnx")?)?;
    /// let api = env.api();
    ///
    /// // One float tensor and one string tensor in the same call.
    /// let mut numbers = vec![1.0f32, 2.0, 3.0];
    /// let outputs = session.run_many(&[
    ///     InputValue::numeric(api, &mut numbers, &[1, 3])?,
    ///     InputValue::strings(api, &["some text"], &[1, 1])?,
    /// ])?;
    /// # let _ = outputs;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_many(&self, inputs: &[InputValue<'_>]) -> Result<Vec<OwnedTensor>> {
        if inputs.len() != self.input_names.len() {
            return Err(Error::IndexOutOfRange {
                kind: "input",
                index: inputs.len(),
                count: self.input_names.len(),
            });
        }
        let input_ptrs: Vec<*const sys::OrtValue> = inputs.iter().map(|v| v.as_ptr()).collect();
        let name_ptrs: Vec<*const c_char> = self.input_names.iter().map(|c| c.as_ptr()).collect();
        self.run_raw(&name_ptrs, &input_ptrs)
    }

    /// Runs the model on inputs supplied by name, in any order.
    ///
    /// # Errors
    ///
    /// Every declared input must appear exactly once. An unknown name, a duplicate, or a
    /// missing input is an error naming the offender.
    pub fn run_named(&self, inputs: &[(&str, InputValue<'_>)]) -> Result<Vec<OwnedTensor>> {
        let declared = || {
            self.input_names()
                .iter()
                .map(|n| format!("{n:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let mut ordered: Vec<*const sys::OrtValue> = Vec::with_capacity(self.input_names.len());
        for declared_name in &self.input_names {
            let expected = declared_name.to_str().unwrap_or_default();
            let mut found: Option<&InputValue<'_>> = None;
            let mut count = 0usize;
            for (name, value) in inputs {
                if *name == expected {
                    count += 1;
                    found = Some(value);
                }
            }
            if count != 1 {
                return Err(Error::InputNotSuppliedOnce {
                    name: expected.to_owned(),
                    count,
                });
            }
            ordered.push(
                found
                    .expect("count == 1 implies a value was found")
                    .as_ptr(),
            );
        }

        for (name, _) in inputs {
            if !self
                .input_names
                .iter()
                .any(|d| d.to_str().unwrap_or_default() == *name)
            {
                return Err(Error::UnknownInput {
                    name: (*name).to_owned(),
                    declared: declared(),
                });
            }
        }

        let name_ptrs: Vec<*const c_char> = self.input_names.iter().map(|c| c.as_ptr()).collect();
        self.run_raw(&name_ptrs, &ordered)
    }

    /// Runs the model on inputs already reduced to raw pointers, requesting every
    /// output.
    ///
    /// `input_names` and `inputs` must be the same length and in the same order. Outputs are
    /// taken into [`OwnedTensor`]s before the run status is checked, so a partial failure
    /// cannot leak the values ONNX Runtime did write.
    ///
    /// `Run` takes `*mut OrtSession`, but ONNX Runtime documents concurrent `Run` on a
    /// shared session as safe, so handing it a pointer derived from `&self` is sound; see
    /// [`Session`].
    fn run_raw(
        &self,
        input_names: &[*const c_char],
        inputs: &[*const sys::OrtValue],
    ) -> Result<Vec<OwnedTensor>> {
        debug_assert_eq!(input_names.len(), inputs.len());
        let run = ort_fn!(self.api, Run);
        let output_name_ptrs: Vec<*const c_char> =
            self.output_names.iter().map(|c| c.as_ptr()).collect();
        let mut output_ptrs: Vec<*mut sys::OrtValue> =
            vec![ptr::null_mut(); self.output_names.len()];

        let status = unsafe {
            run(
                self.ptr,
                ptr::null(),
                input_names.as_ptr(),
                inputs.as_ptr(),
                inputs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                output_ptrs.as_mut_ptr(),
            )
        };

        let outputs: Vec<OwnedTensor> = output_ptrs
            .into_iter()
            .filter(|p| !p.is_null())
            .map(|p| unsafe { OwnedTensor::from_raw(self.api, p) })
            .collect();

        unsafe { self.api.check(status)? };

        Ok(outputs)
    }

    /// Runs the model on the given input tensors.
    ///
    /// Inputs are matched to the model's inputs positionally, and every output
    /// is requested. Returns one [`OwnedTensor`] per model output.
    ///
    /// Every input must hold the same Rust type. Use [`Session::run_many`] or
    /// [`Session::run_named`] for a model whose inputs differ in element type.
    pub fn run<T: Element>(&self, inputs: &[TensorView<'_, T>]) -> Result<Vec<OwnedTensor>> {
        if inputs.len() != self.input_names.len() {
            return Err(Error::IndexOutOfRange {
                kind: "input",
                index: inputs.len(),
                count: self.input_names.len(),
            });
        }
        let input_ptrs: Vec<*const sys::OrtValue> = inputs.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<*const c_char> = self.input_names.iter().map(|c| c.as_ptr()).collect();
        self.run_raw(&name_ptrs, &input_ptrs)
    }

    /// Convenience for the common single-input, single-output case.
    ///
    /// Wraps `data` with `shape`, runs the model, and returns the first output.
    pub fn run_single<T: Element>(&self, data: &mut [T], shape: &[i64]) -> Result<OwnedTensor> {
        let input = TensorView::new(self.api, data, shape)?;
        let mut outputs = self.run(std::slice::from_ref(&input))?;
        if outputs.is_empty() {
            return Err(Error::IndexOutOfRange {
                kind: "output",
                index: 0,
                count: 0,
            });
        }
        Ok(outputs.swap_remove(0))
    }

    /// The shape of input `index`, as reported by the model.
    ///
    /// Symbolic or dynamic dimensions come back as `-1`. A batch axis is
    /// typically dynamic, so a two-dimensional model input usually reports
    /// `[-1, feature_width]`.
    pub fn input_shape(&self, index: usize) -> Result<Vec<i64>> {
        self.io_shape(IoKind::Input, index)
    }

    /// The shape of output `index`, as reported by the model.
    ///
    /// Same convention as [`Self::input_shape`]: `-1` for dynamic dimensions.
    pub fn output_shape(&self, index: usize) -> Result<Vec<i64>> {
        self.io_shape(IoKind::Output, index)
    }

    /// The element type of input `index`.
    pub fn input_type(&self, index: usize) -> Result<DataType> {
        self.io_type(IoKind::Input, index)
    }

    /// The element type of output `index`.
    pub fn output_type(&self, index: usize) -> Result<DataType> {
        self.io_type(IoKind::Output, index)
    }

    /// Everything declared about input `index`: name, element type and shape.
    pub fn input_spec(&self, index: usize) -> Result<TensorSpec> {
        self.io_spec(IoKind::Input, index)
    }

    /// Everything declared about output `index`.
    pub fn output_spec(&self, index: usize) -> Result<TensorSpec> {
        self.io_spec(IoKind::Output, index)
    }

    /// Every input the model declares, in index order.
    pub fn input_specs(&self) -> Result<Vec<TensorSpec>> {
        (0..self.input_names.len())
            .map(|i| self.input_spec(i))
            .collect()
    }

    /// Every output the model declares, in index order.
    pub fn output_specs(&self) -> Result<Vec<TensorSpec>> {
        (0..self.output_names.len())
            .map(|i| self.output_spec(i))
            .collect()
    }

    fn io_spec(&self, kind: IoKind, index: usize) -> Result<TensorSpec> {
        let names = match kind {
            IoKind::Input => &self.input_names,
            IoKind::Output => &self.output_names,
        };
        let name = names
            .get(index)
            .ok_or(Error::IndexOutOfRange {
                kind: kind.as_str(),
                index,
                count: names.len(),
            })?
            .to_str()
            .unwrap_or_default()
            .to_owned();
        Ok(TensorSpec {
            name,
            data_type: self.io_type(kind, index)?,
            shape: self.io_shape(kind, index)?,
        })
    }

    /// Reads a declared element type from the model's type information.
    fn io_type(&self, kind: IoKind, index: usize) -> Result<DataType> {
        let raw = self
            .with_tensor_info(kind, index, |api, info| {
                let f = ort_fn!(api, GetTensorElementType);
                let mut ty: sys::ONNXTensorElementDataType =
                    sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                unsafe { api.check(f(info, &mut ty))? };
                Ok(ty)
            })?
            .ok_or(Error::NotATensor {
                kind: kind.as_str(),
                index,
            })?;
        DataType::from_raw(raw).ok_or(Error::UnsupportedDataType { raw })
    }

    /// Reads a declared shape from the model's type information.
    ///
    /// A non-tensor input or output (a sequence or a map) has no shape and reports an empty
    /// one rather than an error.
    fn io_shape(&self, kind: IoKind, index: usize) -> Result<Vec<i64>> {
        let dims = self.with_tensor_info(kind, index, |api, info| {
            let dim_count_fn = ort_fn!(api, GetDimensionsCount);
            let dims_fn = ort_fn!(api, GetDimensions);

            let mut rank: usize = 0;
            unsafe { api.check(dim_count_fn(info, &mut rank))? };

            let mut dims = vec![0i64; rank];
            if rank > 0 {
                unsafe { api.check(dims_fn(info, dims.as_mut_ptr(), rank))? };
            }
            Ok(dims)
        })?;
        Ok(dims.unwrap_or_default())
    }

    /// Borrows the `OrtTensorTypeAndShapeInfo` for one input or output.
    ///
    /// `SessionGet*TypeInfo` hands back an owned `OrtTypeInfo` that must be released, while
    /// `CastTypeInfoToTensorInfo` *borrows* from it and must not be. Failures inside `f` are
    /// captured rather than propagated so the release is always reached.
    ///
    /// `Ok(None)` means the input or output is not a tensor at all.
    fn with_tensor_info<T, F>(&self, kind: IoKind, index: usize, f: F) -> Result<Option<T>>
    where
        F: FnOnce(Api, *const sys::OrtTensorTypeAndShapeInfo) -> Result<T>,
    {
        let count = match kind {
            IoKind::Input => self.input_names.len(),
            IoKind::Output => self.output_names.len(),
        };
        if index >= count {
            return Err(Error::IndexOutOfRange {
                kind: kind.as_str(),
                index,
                count,
            });
        }

        let type_info_fn = match kind {
            IoKind::Input => ort_fn!(self.api, SessionGetInputTypeInfo),
            IoKind::Output => ort_fn!(self.api, SessionGetOutputTypeInfo),
        };
        let cast_fn = ort_fn!(self.api, CastTypeInfoToTensorInfo);

        let mut type_info: *mut sys::OrtTypeInfo = ptr::null_mut();
        unsafe {
            self.api
                .check(type_info_fn(self.ptr, index, &mut type_info))?
        };

        let result = (|| -> Result<Option<T>> {
            let mut tensor_info: *const sys::OrtTensorTypeAndShapeInfo = ptr::null();
            unsafe { self.api.check(cast_fn(type_info, &mut tensor_info))? };
            if tensor_info.is_null() {
                return Ok(None);
            }
            f(self.api, tensor_info).map(Some)
        })();

        if let Some(release) = self.api.raw().ReleaseTypeInfo {
            unsafe { release(type_info) };
        }
        result
    }

    /// Reads the input or output names out of the session and caches them.
    ///
    /// ONNX Runtime allocates each name with the default allocator and expects
    /// the caller to free it, so each one is copied into a `CString` and the
    /// original released immediately. The allocator itself is owned by ONNX Runtime and
    /// must not be released.
    fn load_io_names(&self, kind: IoKind) -> Result<Vec<CString>> {
        let count_fn = match kind {
            IoKind::Input => ort_fn!(self.api, SessionGetInputCount),
            IoKind::Output => ort_fn!(self.api, SessionGetOutputCount),
        };
        let name_fn = match kind {
            IoKind::Input => ort_fn!(self.api, SessionGetInputName),
            IoKind::Output => ort_fn!(self.api, SessionGetOutputName),
        };
        let allocator_fn = ort_fn!(self.api, GetAllocatorWithDefaultOptions);
        let free_fn = ort_fn!(self.api, AllocatorFree);

        let mut allocator: *mut sys::OrtAllocator = ptr::null_mut();
        unsafe { self.api.check(allocator_fn(&mut allocator))? };

        let mut count: usize = 0;
        unsafe { self.api.check(count_fn(self.ptr, &mut count))? };

        let mut names = Vec::with_capacity(count);
        for index in 0..count {
            let mut raw: *mut c_char = ptr::null_mut();
            unsafe {
                self.api
                    .check(name_fn(self.ptr, index, allocator, &mut raw))?
            };
            if raw.is_null() {
                return Err(Error::IndexOutOfRange {
                    kind: kind.as_str(),
                    index,
                    count,
                });
            }
            let owned = unsafe { CStr::from_ptr(raw).to_owned() };
            unsafe { self.api.check(free_fn(allocator, raw.cast()))? };
            names.push(owned);
        }
        Ok(names)
    }
}

impl Drop for Session {
    /// Releases the session.
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseSession {
            unsafe { release(self.ptr) };
        }
    }
}
