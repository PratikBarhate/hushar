// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Neuron tensors and the tensor sets `nrt_execute` takes.

use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sys;

/// Where a tensor's storage lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    /// In Neuron device memory. Required for execution, and what
    /// [`Model::allocate_io`](crate::Model::allocate_io) allocates.
    #[default]
    Device,
    /// In host memory.
    Host,
}

impl Placement {
    fn to_raw(self) -> sys::nrt_tensor_placement_t {
        match self {
            Self::Device => sys::nrt_tensor_placement_t_NRT_TENSOR_PLACEMENT_DEVICE,
            Self::Host => sys::nrt_tensor_placement_t_NRT_TENSOR_PLACEMENT_HOST,
        }
    }
}

/// A NEFF tensor's element type.
///
/// Values are not contiguous in the C header, so they are mapped explicitly, not cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    BFloat16,
    Float32,
    /// Round-to-nearest float32.
    Float32R,
    Fp8E3,
    Fp8E4,
    Fp8E5,
    /// A dtype this crate does not name, carrying the raw value.
    Unknown(u32),
}

impl DataType {
    pub(crate) fn from_raw(raw: sys::nrt_dtype_t) -> Self {
        match raw {
            sys::nrt_dtype_NRT_DTYPE_UINT8 => Self::Uint8,
            sys::nrt_dtype_NRT_DTYPE_UINT16 => Self::Uint16,
            sys::nrt_dtype_NRT_DTYPE_UINT32 => Self::Uint32,
            sys::nrt_dtype_NRT_DTYPE_UINT64 => Self::Uint64,
            sys::nrt_dtype_NRT_DTYPE_INT8 => Self::Int8,
            sys::nrt_dtype_NRT_DTYPE_INT16 => Self::Int16,
            sys::nrt_dtype_NRT_DTYPE_INT32 => Self::Int32,
            sys::nrt_dtype_NRT_DTYPE_INT64 => Self::Int64,
            sys::nrt_dtype_NRT_DTYPE_FLOAT16 => Self::Float16,
            sys::nrt_dtype_NRT_DTYPE_BFLOAT16 => Self::BFloat16,
            sys::nrt_dtype_NRT_DTYPE_FLOAT32 => Self::Float32,
            sys::nrt_dtype_NRT_DTYPE_FP32R => Self::Float32R,
            sys::nrt_dtype_NRT_DTYPE_FP8_E3 => Self::Fp8E3,
            sys::nrt_dtype_NRT_DTYPE_FP8_E4 => Self::Fp8E4,
            sys::nrt_dtype_NRT_DTYPE_FP8_E5 => Self::Fp8E5,
            other => Self::Unknown(other),
        }
    }

    /// Size of one element in bytes, or `None` for an unrecognised dtype.
    pub fn size_of(self) -> Option<usize> {
        Some(match self {
            Self::Uint8 | Self::Int8 | Self::Fp8E3 | Self::Fp8E4 | Self::Fp8E5 => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 | Self::Float32R => 4,
            Self::Uint64 | Self::Int64 => 8,
            Self::Unknown(_) => return None,
        })
    }
}

/// A tensor owned by the Neuron runtime.
///
/// Holds an `Arc<Runtime>` so the runtime cannot be closed underneath it, and frees the
/// libnrt allocation on drop.
///
/// `Send` is sound because a tensor is an opaque libnrt allocation reached only through
/// libnrt calls. It is deliberately not `Sync`: the `&mut self` on the write methods is
/// what prevents two threads mutating the same tensor at once.
pub struct Tensor {
    runtime: Arc<Runtime>,
    ptr: *mut sys::nrt_tensor_t,
    name: String,
}

unsafe impl Send for Tensor {}

impl Tensor {
    /// Allocates a tensor of `size` bytes on the given NeuronCore.
    pub fn allocate(
        runtime: Arc<Runtime>,
        name: &str,
        size: usize,
        placement: Placement,
        neuron_core: i32,
    ) -> Result<Self> {
        let c_name = CString::new(name).map_err(|_| Error::InteriorNul {
            context: "tensor name",
        })?;
        let mut ptr: *mut sys::nrt_tensor_t = ptr::null_mut();

        unsafe {
            let status = (runtime.symbols.nrt_tensor_allocate)(
                placement.to_raw(),
                neuron_core,
                size,
                c_name.as_ptr(),
                &mut ptr,
            );
            runtime.symbols.check("nrt_tensor_allocate", status)?;
        }

        Ok(Self {
            runtime,
            ptr,
            name: name.to_owned(),
        })
    }

    /// The tensor's name, as given to the model or at allocation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tensor's size in bytes, as libnrt reports it.
    pub fn size(&self) -> usize {
        unsafe { (self.runtime.symbols.nrt_tensor_get_size)(self.ptr) }
    }

    /// Copies `data` into the tensor at `offset` bytes.
    ///
    /// # Errors
    ///
    /// [`Error::SizeMismatch`] if the write would run past the end of the tensor.
    pub fn write_bytes(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let size = self.size();
        if offset.saturating_add(data.len()) > size {
            return Err(Error::SizeMismatch {
                name: self.name.clone(),
                tensor_bytes: size,
                buffer_bytes: offset + data.len(),
            });
        }

        unsafe {
            let status = (self.runtime.symbols.nrt_tensor_write)(
                self.ptr,
                data.as_ptr().cast::<std::os::raw::c_void>(),
                offset,
                data.len(),
            );
            self.runtime.symbols.check("nrt_tensor_write", status)
        }
    }

    /// Copies the whole tensor into a freshly allocated byte vector.
    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        let size = self.size();
        let mut buf = vec![0u8; size];
        self.read_bytes_into(0, &mut buf)?;
        Ok(buf)
    }

    /// Copies `buf.len()` bytes from `offset` into `buf`.
    pub fn read_bytes_into(&self, offset: usize, buf: &mut [u8]) -> Result<()> {
        let size = self.size();
        if offset.saturating_add(buf.len()) > size {
            return Err(Error::SizeMismatch {
                name: self.name.clone(),
                tensor_bytes: size,
                buffer_bytes: offset + buf.len(),
            });
        }

        unsafe {
            let status = (self.runtime.symbols.nrt_tensor_read)(
                self.ptr,
                buf.as_mut_ptr().cast::<std::os::raw::c_void>(),
                offset,
                buf.len(),
            );
            self.runtime.symbols.check("nrt_tensor_read", status)
        }
    }

    /// Copies a slice of `f32` into the tensor, starting at byte 0.
    ///
    /// `f32` has no padding or invalid bit patterns, so viewing the slice as bytes is
    /// sound.
    pub fn write_f32(&mut self, data: &[f32]) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        self.write_bytes(0, bytes)
    }

    /// Reads the tensor as `f32` values.
    ///
    /// # Errors
    ///
    /// [`Error::SizeMismatch`] if the tensor's size is not a whole number of `f32`s.
    pub fn read_f32(&self) -> Result<Vec<f32>> {
        let size = self.size();
        if !size.is_multiple_of(std::mem::size_of::<f32>()) {
            return Err(Error::SizeMismatch {
                name: self.name.clone(),
                tensor_bytes: size,
                buffer_bytes: size.next_multiple_of(std::mem::size_of::<f32>()),
            });
        }

        let mut values = vec![0f32; size / std::mem::size_of::<f32>()];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), size) };
        self.read_bytes_into(0, bytes)?;
        Ok(values)
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::nrt_tensor_t {
        self.ptr
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("name", &self.name)
            .field("size", &self.size())
            .finish()
    }
}

impl Drop for Tensor {
    /// Frees the libnrt tensor allocation.
    ///
    /// `nrt_tensor_free` takes a pointer to the handle and nulls it, so the allocation is
    /// released exactly once.
    fn drop(&mut self) {
        unsafe { (self.runtime.symbols.nrt_tensor_free)(&mut self.ptr) };
    }
}

/// The named bundle of tensors `nrt_execute` reads inputs from and writes outputs to.
///
/// The libnrt set only refers to its tensors; this type holds them, so a tensor cannot be
/// dropped while the set still points at it.
///
/// `Send` is sound because the set and the tensors it refers to are opaque libnrt
/// allocations reached only through libnrt calls, and mutated only behind `&mut self`.
pub struct TensorSet {
    runtime: Arc<Runtime>,
    ptr: *mut sys::nrt_tensor_set_t,
    tensors: Vec<Tensor>,
}

unsafe impl Send for TensorSet {}

impl TensorSet {
    /// Creates an empty tensor set.
    pub fn new(runtime: Arc<Runtime>) -> Result<Self> {
        let mut ptr: *mut sys::nrt_tensor_set_t = ptr::null_mut();
        unsafe {
            let status = (runtime.symbols.nrt_allocate_tensor_set)(&mut ptr);
            runtime.symbols.check("nrt_allocate_tensor_set", status)?;
        }
        Ok(Self {
            runtime,
            ptr,
            tensors: Vec::new(),
        })
    }

    /// Adds `tensor` under `name`, transferring ownership to the set.
    ///
    /// `name` must match the NEFF's tensor name; that is how libnrt binds the buffer to a
    /// graph input or output. libnrt stores the tensor pointer, so the tensor is not
    /// dropped until after the set is destroyed.
    pub fn insert(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        let c_name = CString::new(name).map_err(|_| Error::InteriorNul {
            context: "tensor set entry name",
        })?;

        unsafe {
            let status = (self.runtime.symbols.nrt_add_tensor_to_tensor_set)(
                self.ptr,
                c_name.as_ptr(),
                tensor.as_ptr(),
            );
            self.runtime
                .symbols
                .check("nrt_add_tensor_to_tensor_set", status)?;
        }

        self.tensors.push(tensor);
        Ok(())
    }

    /// The tensors in this set, in insertion order.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// Looks up a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name() == name)
    }

    /// Looks up a tensor by name for mutation.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.tensors.iter_mut().find(|t| t.name() == name)
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::nrt_tensor_set_t {
        self.ptr
    }
}

impl std::fmt::Debug for TensorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorSet")
            .field("tensors", &self.tensors)
            .finish()
    }
}

impl Drop for TensorSet {
    /// Destroys the libnrt set.
    ///
    /// Runs before `self.tensors` is dropped, so the set never refers to freed tensors.
    /// `nrt_destroy_tensor_set` takes a pointer to the handle and nulls it, so the set is
    /// destroyed exactly once.
    fn drop(&mut self) {
        unsafe { (self.runtime.symbols.nrt_destroy_tensor_set)(&mut self.ptr) };
    }
}
