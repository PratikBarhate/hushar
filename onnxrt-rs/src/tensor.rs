// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Tensors: `OrtValue` wrappers plus the element-type mapping.

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::ptr;

use crate::api::{Api, ort_fn};
use crate::error::{Error, Result};
use crate::sys;

/// A Rust type that can be stored in an ONNX tensor.
///
/// # Safety
///
/// `ELEMENT_TYPE` must be the `ONNXTensorElementDataType` whose in-memory
/// representation is exactly `Self`. Getting this wrong lets ONNX Runtime
/// reinterpret bytes as the wrong type.
pub unsafe trait Element: Copy + 'static {
    /// The ONNX element type tag for `Self`.
    const ELEMENT_TYPE: sys::ONNXTensorElementDataType;
    /// Human-readable name, used in error messages.
    const NAME: &'static str;
}

macro_rules! impl_element {
    ($rust:ty => $tag:expr, $name:literal) => {
        // SAFETY: the tag on the right is the ONNX element type whose layout is
        // identical to the Rust type on the left; both are verified against the
        // vendored header.
        unsafe impl Element for $rust {
            const ELEMENT_TYPE: sys::ONNXTensorElementDataType = $tag;
            const NAME: &'static str = $name;
        }
    };
}

impl_element!(f32 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "f32");
impl_element!(f64 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "f64");
impl_element!(i8  => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "i8");
impl_element!(u8  => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "u8");
impl_element!(i16 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "i16");
impl_element!(u16 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "u16");
impl_element!(i32 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "i32");
impl_element!(u32 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "u32");
impl_element!(i64 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "i64");
impl_element!(u64 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "u64");

// `half`'s types are `repr(transparent)` over `u16`, which is exactly ONNX's
// FLOAT16 and BFLOAT16 storage layout, so tensors of them are zero-copy like any
// other numeric type. `half` computes by widening to `f32`, but nothing here
// computes: these are transported and the engine does the arithmetic.
impl_element!(half::f16 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "f16");
impl_element!(half::bf16 => sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, "bf16");

// `bool` is deliberately absent, and it is the one gap worth explaining.
//
// ONNX BOOL is one byte, as is Rust's `bool`, so the layout matches and an
// `impl Element for bool` would compile and appear to work. But Rust's `bool` has
// only two valid bit patterns, and `as_slice::<bool>()` would hand out a
// `&[bool]` over runtime-written memory: a producer that wrote `2` into a BOOL
// tensor would make that slice undefined behaviour, not merely wrong. The ONNX
// specification does require 0 or 1, so this would probably never bite — which is
// exactly what makes it a bad thing to rely on.
//
// Booleans are therefore reached through [`InputValue::bools`] and
// [`OwnedTensor::bools`], which cross the boundary as `u8` and convert. Writing
// is sound because Rust produces the bytes; reading is sound because a non-zero
// byte becomes `true` rather than an invalid `bool`.

/// An ONNX tensor element type, named rather than numeric.
///
/// [`Element`] covers the types that can be borrowed as a Rust slice; this
/// covers every type that can appear on a model's inputs or outputs, including
/// the two that cannot be a plain slice — `String`, whose elements are
/// separately allocated, and `Bool`, for the reason given above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    /// UTF-8 strings. Each element is separately allocated by the runtime.
    String,
}

impl DataType {
    /// Maps an ONNX element type tag, or `None` for one this crate does not
    /// model — the 8-bit and 4-bit float formats, complex numbers, and the
    /// sub-byte integers.
    // `bindgen` names these constants after the C enum, which is not
    // `UPPER_CASE`, and a constant in a pattern is expected to be. Matching on
    // the generated names is still the right thing to do -- the alternative is
    // matching bare integers, which would silently break if a tag ever moved.
    #[allow(non_upper_case_globals)]
    pub fn from_raw(raw: sys::ONNXTensorElementDataType) -> Option<Self> {
        use sys::*;
        Some(match raw {
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Self::Bool,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => Self::I8,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => Self::I16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Self::I32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Self::I64,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Self::U8,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => Self::U16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Self::U32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => Self::U64,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Self::F16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => Self::Bf16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Self::F32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Self::F64,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => Self::String,
            _ => return None,
        })
    }

    /// The ONNX element type tag for this type.
    pub fn to_raw(self) -> sys::ONNXTensorElementDataType {
        use sys::*;
        match self {
            Self::Bool => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            Self::I8 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            Self::I16 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
            Self::I32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            Self::I64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            Self::U8 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            Self::U16 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            Self::U32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            Self::U64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            Self::F16 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            Self::Bf16 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
            Self::F32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Self::F64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            Self::String => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
        }
    }

    /// Bytes per element, or `None` for [`DataType::String`], whose elements are
    /// variable length.
    ///
    /// This is what a caller needs to split a raw byte payload into elements —
    /// the representation the Open Inference Protocol requires for 16-bit
    /// floats, which have no protobuf field type.
    pub fn size_bytes(self) -> Option<usize> {
        Some(match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::String => return None,
        })
    }

    /// The name used in errors and in the wire format.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::String => "string",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// How many elements `shape` describes, or `None` if it overflows or holds a
/// negative dimension.
///
/// A model's declared shape may contain `-1` for a dynamic axis, but a shape
/// being handed to the runtime alongside a buffer must be fully concrete.
fn element_count(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        if d < 0 {
            None
        } else {
            acc.checked_mul(d as usize)
        }
    })
}

/// CPU memory description, needed to build tensors over host buffers.
#[derive(Debug)]
pub(crate) struct MemoryInfo {
    api: Api,
    ptr: *mut sys::OrtMemoryInfo,
}

impl MemoryInfo {
    pub(crate) fn cpu(api: Api) -> Result<Self> {
        let create = ort_fn!(api, CreateCpuMemoryInfo);
        let mut ptr: *mut sys::OrtMemoryInfo = ptr::null_mut();
        // SAFETY: `ptr` is a valid out-parameter; the arena allocator and default
        // memory type are the documented values for host-visible CPU buffers.
        unsafe {
            api.check(create(
                sys::OrtAllocatorType_OrtArenaAllocator,
                sys::OrtMemType_OrtMemTypeDefault,
                &mut ptr,
            ))?;
        }
        Ok(Self { api, ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const sys::OrtMemoryInfo {
        self.ptr
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseMemoryInfo {
            // SAFETY: created by `CreateCpuMemoryInfo`, released exactly once.
            unsafe { release(self.ptr) };
        }
    }
}

/// A tensor built over a caller-owned buffer.
///
/// ONNX Runtime does not copy the data, so the buffer must stay alive and
/// unmoved for as long as the tensor exists. The `'data` lifetime encodes that:
/// the borrow checker will not let the tensor outlive the slice it points at.
#[derive(Debug)]
pub struct TensorView<'data, T: Element> {
    api: Api,
    ptr: *mut sys::OrtValue,
    // Keeps `MemoryInfo` alive alongside the value it described.
    _memory_info: MemoryInfo,
    _data: PhantomData<&'data mut [T]>,
}

impl<'data, T: Element> TensorView<'data, T> {
    /// Wraps `data` as a tensor of the given `shape` without copying.
    ///
    /// `shape` must describe exactly `data.len()` elements.
    ///
    /// The slice is taken by mutable reference because ONNX Runtime's
    /// `CreateTensorWithDataAsOrtValue` takes a non-const pointer; the data is
    /// not modified when the tensor is used as an input.
    pub fn new(api: Api, data: &'data mut [T], shape: &[i64]) -> Result<Self> {
        check_shape(shape, data.len())?;

        let memory_info = MemoryInfo::cpu(api)?;
        let create = ort_fn!(api, CreateTensorWithDataAsOrtValue);
        let mut ptr: *mut sys::OrtValue = ptr::null_mut();

        // SAFETY: `data` is a valid, uniquely borrowed slice of `T` whose length
        // in bytes we compute exactly; `T::ELEMENT_TYPE` matches `T`'s layout by
        // the `Element` contract; `shape` is a valid slice of `shape.len()`
        // dimensions. The `'data` lifetime on `Self` keeps the buffer borrowed
        // for as long as the returned tensor lives, which is what ONNX Runtime
        // requires of a non-owning tensor.
        unsafe {
            api.check(create(
                memory_info.as_ptr(),
                data.as_mut_ptr().cast::<std::os::raw::c_void>(),
                std::mem::size_of_val(data),
                shape.as_ptr(),
                shape.len(),
                T::ELEMENT_TYPE,
                &mut ptr,
            ))?;
        }

        Ok(Self {
            api,
            ptr,
            _memory_info: memory_info,
            _data: PhantomData,
        })
    }

    pub(crate) fn as_ptr(&self) -> *const sys::OrtValue {
        self.ptr
    }

    /// Erases the element type, giving a value that can sit in a mixed batch of
    /// inputs.
    ///
    /// The borrow on the underlying buffer is preserved, so this is free: no
    /// copy, and the same aliasing guarantee. See [`InputValue`] for why the
    /// erasure happens here rather than in the type system.
    pub fn into_input(self) -> InputValue<'data> {
        // Move the two owned resources out without running `Drop`, which would
        // release the very `OrtValue` that `InputValue` is taking over.
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is a `ManuallyDrop`, so its destructor never runs and
        // reading the field out transfers ownership exactly once. `api` and `ptr`
        // are `Copy`; `MemoryInfo` is not, hence the explicit read.
        let memory_info = unsafe { ptr::read(&this._memory_info) };
        InputValue {
            api: this.api,
            ptr: this.ptr,
            _memory_info: Some(memory_info),
            _data: PhantomData,
        }
    }
}

impl<T: Element> Drop for TensorView<'_, T> {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseValue {
            // SAFETY: created by `CreateTensorWithDataAsOrtValue`, released once.
            // Releasing the value does not free the caller's buffer.
            unsafe { release(self.ptr) };
        }
    }
}

/// A tensor prepared as a model input, with its element type erased.
///
/// # Why erase here
///
/// [`Session::run`](crate::Session::run) is generic over one [`Element`], so it
/// is multi-input but *homogeneous*: every tensor in the call must hold the same
/// Rust type. A model taking a float feature vector alongside a string vector —
/// an embedding model with raw text, say — cannot be expressed that way at any
/// number of supported dtypes.
///
/// The fix is to stop tracking the element type once the tensor exists. ONNX
/// Runtime's `Run` takes an array of `OrtValue*` and reads each value's type from
/// the value itself, so the type parameter has no work left to do by then.
/// Erasing at this boundary rather than with an enum over `TensorView<T>` keeps
/// [`Element`] as the extension point — a new dtype is one `impl`, not a new
/// variant threaded through every match — and keeps numeric tensors zero-copy.
///
/// # Lifetime
///
/// `'data` is the borrow on the caller's buffer. Numeric tensors point straight
/// at it, so it must outlive the value. String tensors own runtime-allocated
/// memory and borrow nothing, so [`InputValue::strings`] returns `'static`, which
/// coerces into any shorter lifetime and can therefore sit in the same slice as
/// borrowed tensors.
#[derive(Debug)]
pub struct InputValue<'data> {
    api: Api,
    ptr: *mut sys::OrtValue,
    /// Held for tensors built over borrowed host memory, so the memory
    /// description outlives the value that referred to it. `None` for tensors the
    /// runtime allocated, which need no such description.
    _memory_info: Option<MemoryInfo>,
    _data: PhantomData<&'data mut ()>,
}

impl<'data> InputValue<'data> {
    /// Wraps `data` as a tensor of `shape` without copying.
    ///
    /// Equivalent to `TensorView::new(..)?.into_input()`, which is the more
    /// direct spelling when the typed view is not needed for anything else.
    pub fn numeric<T: Element>(api: Api, data: &'data mut [T], shape: &[i64]) -> Result<Self> {
        Ok(TensorView::new(api, data, shape)?.into_input())
    }

    pub(crate) fn as_ptr(&self) -> *const sys::OrtValue {
        self.ptr
    }
}

impl InputValue<'static> {
    /// Builds a `BOOL` tensor from `values`.
    ///
    /// Copies, because it also converts. `bool` is not an [`Element`] — see the
    /// note where the impls are defined — so the bytes are written into
    /// runtime-allocated storage rather than borrowed from the caller.
    pub fn bools(api: Api, values: &[bool], shape: &[i64]) -> Result<Self> {
        check_shape(shape, values.len())?;
        let value = Self::allocated(api, shape, DataType::Bool)?;
        if !values.is_empty() {
            let get_data = ort_fn!(api, GetTensorMutableData);
            let mut data: *mut std::os::raw::c_void = ptr::null_mut();
            // SAFETY: `value.ptr` is a live tensor; `data` is a valid
            // out-parameter. The runtime writes a pointer to its own buffer,
            // which is `values.len()` bytes because BOOL is one byte per element
            // and the shape was checked to match.
            unsafe { api.check(get_data(value.ptr, &mut data))? };
            debug_assert!(!data.is_null(), "non-empty tensor has a null data buffer");
            // SAFETY: the destination holds exactly `values.len()` bytes, one per
            // element of a BOOL tensor of this shape. Writing 0 or 1 per element
            // is the representation ONNX requires.
            unsafe {
                for (i, &b) in values.iter().enumerate() {
                    data.cast::<u8>().add(i).write(u8::from(b));
                }
            }
        }
        Ok(value)
    }

    /// Builds a `STRING` tensor from `values`, laid out row-major for `shape`.
    ///
    /// **This copies, and there is no zero-copy alternative.** ONNX Runtime
    /// builds a string tensor with `CreateTensorAsOrtValue` and then
    /// `FillStringTensor`, which duplicates each string into runtime-owned
    /// memory. The C API has no `CreateTensorWithDataAsOrtValue` equivalent for
    /// strings, because a string tensor is an array of separately allocated
    /// pointers rather than one contiguous block. Recorded here so the absence
    /// looks deliberate rather than missing.
    pub fn strings<S: AsRef<str>>(api: Api, values: &[S], shape: &[i64]) -> Result<Self> {
        check_shape(shape, values.len())?;

        // Reject interior NULs up front: `CString::new` would otherwise fail
        // partway through, and a caller wants to know which input was bad.
        let owned: Vec<CString> = values
            .iter()
            .map(|s| {
                CString::new(s.as_ref()).map_err(|_| Error::InteriorNul {
                    context: "string tensor element",
                })
            })
            .collect::<Result<_>>()?;

        let value = Self::allocated(api, shape, DataType::String)?;
        if !owned.is_empty() {
            let raw: Vec<*const c_char> = owned.iter().map(|c| c.as_ptr()).collect();
            let fill = ort_fn!(api, FillStringTensor);
            // SAFETY: `value.ptr` is a live STRING tensor with exactly
            // `raw.len()` elements, and `raw` holds that many NUL-terminated
            // pointers, live for the call because `owned` outlives it.
            // `FillStringTensor` copies each string, so the pointers are not
            // retained. On failure `value`'s `Drop` releases the tensor.
            unsafe { api.check(fill(value.ptr, raw.as_ptr(), raw.len()))? };
        }

        Ok(value)
    }

    /// An empty tensor of `data_type` and `shape`, with storage allocated by the
    /// runtime rather than borrowed from the caller.
    fn allocated(api: Api, shape: &[i64], data_type: DataType) -> Result<Self> {
        let allocator = default_allocator(api)?;
        let create = ort_fn!(api, CreateTensorAsOrtValue);
        let mut ptr: *mut sys::OrtValue = ptr::null_mut();
        // SAFETY: `allocator` is ONNX Runtime's default allocator, `shape` is a
        // valid slice of the length passed alongside it, and `ptr` is a valid
        // out-parameter. The runtime allocates the tensor's storage, so nothing
        // of ours needs to stay alive for it — hence the `'static` lifetime.
        unsafe {
            api.check(create(
                allocator,
                shape.as_ptr(),
                shape.len(),
                data_type.to_raw(),
                &mut ptr,
            ))?;
        }
        Ok(Self {
            api,
            ptr,
            _memory_info: None,
            _data: PhantomData,
        })
    }
}

/// Fails unless `shape` describes exactly `actual` elements.
fn check_shape(shape: &[i64], actual: usize) -> Result<()> {
    let expected = element_count(shape).ok_or_else(|| Error::ShapeMismatch {
        shape: shape.to_vec(),
        expected: 0,
        actual,
    })?;
    if expected != actual {
        return Err(Error::ShapeMismatch {
            shape: shape.to_vec(),
            expected,
            actual,
        });
    }
    Ok(())
}

impl Drop for InputValue<'_> {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseValue {
            // SAFETY: this type owns the value, whether it came from a
            // `TensorView` or was allocated here, and releases it exactly once.
            // For a borrowed tensor this does not free the caller's buffer.
            unsafe { release(self.ptr) };
        }
    }
}

/// ONNX Runtime's process-wide default allocator.
///
/// Owned by the runtime, so it must not be released.
fn default_allocator(api: Api) -> Result<*mut sys::OrtAllocator> {
    let f = ort_fn!(api, GetAllocatorWithDefaultOptions);
    let mut allocator: *mut sys::OrtAllocator = ptr::null_mut();
    // SAFETY: valid out-parameter; the returned allocator is a static owned by
    // ONNX Runtime.
    unsafe { api.check(f(&mut allocator))? };
    Ok(allocator)
}

/// A tensor produced by the runtime, which owns its buffer.
#[derive(Debug)]
pub struct OwnedTensor {
    api: Api,
    ptr: *mut sys::OrtValue,
}

impl OwnedTensor {
    /// Takes ownership of a non-null `OrtValue`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a non-null `OrtValue` produced by `api` that nothing else
    /// will release.
    pub(crate) unsafe fn from_raw(api: Api, ptr: *mut sys::OrtValue) -> Self {
        Self { api, ptr }
    }

    /// The tensor's shape.
    pub fn shape(&self) -> Result<Vec<i64>> {
        let info = self.type_and_shape()?;
        info.shape()
    }

    /// The number of elements in the tensor.
    pub fn len(&self) -> Result<usize> {
        let shape = self.shape()?;
        shape.iter().try_fold(1usize, |acc, &d| {
            if d < 0 {
                Err(Error::NonConcreteDimension {
                    shape: shape.clone(),
                    dim: d,
                })
            } else {
                Ok(acc * d as usize)
            }
        })
    }

    /// Whether the tensor holds no elements.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Borrows the tensor's contents as a slice of `T`.
    ///
    /// Fails if the tensor's element type is not `T`, so a wrong guess is a
    /// clean error rather than reinterpreted bytes.
    pub fn as_slice<T: Element>(&self) -> Result<&[T]> {
        let info = self.type_and_shape()?;
        let actual = info.element_type()?;
        if actual != T::ELEMENT_TYPE {
            return Err(Error::ElementTypeMismatch {
                requested: T::NAME,
                actual,
            });
        }

        let len = self.len()?;
        let get_data = ort_fn!(self.api, GetTensorMutableData);
        let mut data: *mut std::os::raw::c_void = ptr::null_mut();
        // SAFETY: `ptr` is a live tensor `OrtValue`; `data` is a valid
        // out-parameter. ONNX Runtime writes a pointer to the tensor's buffer,
        // which stays valid while `self` is alive.
        unsafe {
            self.api.check(get_data(self.ptr, &mut data))?;
        }

        if len == 0 {
            return Ok(&[]);
        }
        debug_assert!(!data.is_null(), "non-empty tensor has a null data pointer");

        // SAFETY: the element type was just checked to be `T`, `len` is the
        // element count from the same tensor's shape, and the buffer is owned by
        // `self`, so the returned slice cannot outlive it.
        Ok(unsafe { std::slice::from_raw_parts(data.cast::<T>(), len) })
    }

    /// Copies the tensor's contents into a `Vec`.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        Ok(self.as_slice::<T>()?.to_vec())
    }

    /// The tensor's element type.
    ///
    /// Fails with [`Error::UnsupportedDataType`] for the element types this crate
    /// does not model, so an exotic model reports what it holds rather than
    /// producing a wrong answer.
    pub fn data_type(&self) -> Result<DataType> {
        let raw = self.type_and_shape()?.element_type()?;
        DataType::from_raw(raw).ok_or(Error::UnsupportedDataType { raw })
    }

    /// Reads a `STRING` tensor's contents.
    ///
    /// Always copies. String elements are separately allocated inside the
    /// runtime, so there is nothing contiguous to borrow — the mirror of the note
    /// on [`InputValue::strings`].
    ///
    /// Invalid UTF-8 is replaced rather than rejected: a model that emits one bad
    /// byte should not fail a whole batch.
    pub fn strings(&self) -> Result<Vec<String>> {
        let actual = self.data_type()?;
        if actual != DataType::String {
            return Err(Error::ElementTypeMismatch {
                requested: "string",
                actual: actual.to_raw(),
            });
        }

        let count = self.len()?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let length_fn = ort_fn!(self.api, GetStringTensorDataLength);
        let content_fn = ort_fn!(self.api, GetStringTensorContent);

        let mut total: usize = 0;
        // SAFETY: live tensor, valid out-parameter. Returns the summed byte
        // length of every element, excluding NUL terminators.
        unsafe { self.api.check(length_fn(self.ptr, &mut total))? };

        let mut bytes = vec![0u8; total];
        let mut offsets = vec![0usize; count];
        // SAFETY: `bytes` has room for exactly `total` bytes and `offsets` for
        // exactly `count` entries, which are the two lengths passed alongside
        // them, so ONNX Runtime cannot write out of bounds.
        unsafe {
            self.api.check(content_fn(
                self.ptr,
                bytes.as_mut_ptr().cast::<std::os::raw::c_void>(),
                total,
                offsets.as_mut_ptr(),
                count,
            ))?
        };

        // `offsets[i]` starts element `i`; it ends where the next one starts, or
        // at `total` for the last. The bounds are checked rather than trusted:
        // slicing on a bad offset would panic inside a request handler.
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let start = offsets[i];
            let end = offsets.get(i + 1).copied().unwrap_or(total);
            if start > end || end > total {
                return Err(Error::MalformedStringTensor {
                    index: i,
                    start,
                    end,
                    total,
                });
            }
            out.push(String::from_utf8_lossy(&bytes[start..end]).into_owned());
        }
        Ok(out)
    }

    /// Reads a `BOOL` tensor's contents.
    ///
    /// Goes through `u8` so that a byte outside `{0, 1}` becomes `true` rather
    /// than an invalid `bool`. See the note on the [`Element`] impls.
    pub fn bools(&self) -> Result<Vec<bool>> {
        let actual = self.data_type()?;
        if actual != DataType::Bool {
            return Err(Error::ElementTypeMismatch {
                requested: "bool",
                actual: actual.to_raw(),
            });
        }

        let count = self.len()?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let get_data = ort_fn!(self.api, GetTensorMutableData);
        let mut data: *mut std::os::raw::c_void = ptr::null_mut();
        // SAFETY: live tensor, valid out-parameter.
        unsafe { self.api.check(get_data(self.ptr, &mut data))? };
        debug_assert!(!data.is_null(), "non-empty tensor has a null data buffer");

        // SAFETY: a BOOL tensor stores one byte per element, and `count` is the
        // element count from this tensor's own shape. Read as `u8`, which has no
        // invalid bit patterns, so a non-conforming value cannot be unsound.
        let raw = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), count) };
        Ok(raw.iter().map(|&b| b != 0).collect())
    }

    fn type_and_shape(&self) -> Result<TypeAndShape> {
        let get = ort_fn!(self.api, GetTensorTypeAndShape);
        let mut ptr: *mut sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
        // SAFETY: `self.ptr` is a live tensor and `ptr` a valid out-parameter.
        unsafe {
            self.api.check(get(self.ptr, &mut ptr))?;
        }
        Ok(TypeAndShape { api: self.api, ptr })
    }
}

impl Drop for OwnedTensor {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseValue {
            // SAFETY: owned per `from_raw`'s contract, released exactly once.
            unsafe { release(self.ptr) };
        }
    }
}

/// Owned `OrtTensorTypeAndShapeInfo`.
struct TypeAndShape {
    api: Api,
    ptr: *mut sys::OrtTensorTypeAndShapeInfo,
}

impl TypeAndShape {
    fn shape(&self) -> Result<Vec<i64>> {
        let count_fn = ort_fn!(self.api, GetDimensionsCount);
        let dims_fn = ort_fn!(self.api, GetDimensions);

        let mut ndim: usize = 0;
        // SAFETY: live info handle, valid out-parameter.
        unsafe { self.api.check(count_fn(self.ptr, &mut ndim))? };

        let mut dims = vec![0i64; ndim];
        if ndim > 0 {
            // SAFETY: `dims` has exactly `ndim` slots, which is the length we
            // pass, so ONNX Runtime cannot write out of bounds.
            unsafe { self.api.check(dims_fn(self.ptr, dims.as_mut_ptr(), ndim))? };
        }
        Ok(dims)
    }

    fn element_type(&self) -> Result<sys::ONNXTensorElementDataType> {
        let f = ort_fn!(self.api, GetTensorElementType);
        let mut ty: sys::ONNXTensorElementDataType =
            sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        // SAFETY: live info handle, valid out-parameter.
        unsafe { self.api.check(f(self.ptr, &mut ty))? };
        Ok(ty)
    }
}

impl Drop for TypeAndShape {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw().ReleaseTensorTypeAndShapeInfo {
            // SAFETY: created by `GetTensorTypeAndShape`, released exactly once.
            unsafe { release(self.ptr) };
        }
    }
}
