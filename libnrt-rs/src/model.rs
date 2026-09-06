// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Loading a compiled NEFF and executing it.

use std::ptr;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::runtime::{Runtime, cstr_from_array};
use crate::sys;
use crate::tensor::{DataType, Placement, Tensor, TensorSet};

/// Whether a model tensor is an input or an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Usage {
    Input,
    Output,
}

impl Usage {
    fn from_raw(raw: sys::nrt_tensor_usage_t) -> Self {
        if raw == sys::nrt_tensor_usage_NRT_TENSOR_USAGE_INPUT {
            Self::Input
        } else {
            Self::Output
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }
}

/// What a NEFF says about one of its tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub usage: Usage,
    /// Size in bytes.
    pub size: usize,
    pub dtype: DataType,
    pub shape: Vec<u32>,
}

/// A NEFF loaded onto one or more NeuronCores.
///
/// Holds an `Arc<Runtime>`, so the runtime stays initialised for as long as the
/// model is loaded.
pub struct Model {
    runtime: Arc<Runtime>,
    ptr: *mut sys::nrt_model_t,
    tensors: Vec<TensorInfo>,
}

// SAFETY: `nrt_execute` may be called concurrently on a loaded model, and this
// type exposes only `&self` accessors plus `execute`, whose mutable state lives
// entirely in the caller's `TensorSet`.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Loads a compiled NEFF onto `neuron_core_count` cores starting at
    /// `start_neuron_core`.
    ///
    /// Pass `start_neuron_core = 0` and `neuron_core_count = 1` for the common
    /// single-core case. `neuron_core_count` must match what the NEFF was
    /// compiled for, or the load fails with
    /// [`Status::NotEnoughNeuronCores`](crate::Status::NotEnoughNeuronCores).
    pub fn load(
        runtime: Arc<Runtime>,
        neff: &[u8],
        start_neuron_core: i32,
        neuron_core_count: i32,
    ) -> Result<Self> {
        let mut ptr: *mut sys::nrt_model_t = ptr::null_mut();

        // SAFETY: `neff` is valid for the length we pass, and `ptr` is a valid
        // out-parameter. libnrt copies what it needs during the call, so the
        // borrow does not have to outlive it.
        unsafe {
            let status = (runtime.symbols.nrt_load)(
                neff.as_ptr().cast::<std::os::raw::c_void>(),
                neff.len(),
                start_neuron_core,
                neuron_core_count,
                &mut ptr,
            );
            runtime.symbols.check("nrt_load", status)?;
        }

        let mut model = Self {
            runtime,
            ptr,
            tensors: Vec::new(),
        };
        model.tensors = model.read_tensor_info()?;
        Ok(model)
    }

    /// Everything the NEFF declares about its tensors.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// The model's input tensors.
    pub fn inputs(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.iter().filter(|t| t.usage == Usage::Input)
    }

    /// The model's output tensors.
    pub fn outputs(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.iter().filter(|t| t.usage == Usage::Output)
    }

    /// Allocates a device tensor for every input and every output, sized from
    /// the NEFF.
    ///
    /// Returns `(inputs, outputs)` ready to hand to [`Model::execute`]. Write
    /// features into the input set, execute, then read the output set.
    pub fn allocate_io(&self, neuron_core: i32) -> Result<(TensorSet, TensorSet)> {
        let mut inputs = TensorSet::new(Arc::clone(&self.runtime))?;
        for info in self.inputs() {
            let tensor = Tensor::allocate(
                Arc::clone(&self.runtime),
                &info.name,
                info.size,
                Placement::Device,
                neuron_core,
            )?;
            inputs.insert(&info.name, tensor)?;
        }

        let mut outputs = TensorSet::new(Arc::clone(&self.runtime))?;
        for info in self.outputs() {
            let tensor = Tensor::allocate(
                Arc::clone(&self.runtime),
                &info.name,
                info.size,
                Placement::Device,
                neuron_core,
            )?;
            outputs.insert(&info.name, tensor)?;
        }

        Ok((inputs, outputs))
    }

    /// Runs the model, reading from `inputs` and writing into `outputs`.
    ///
    /// `outputs` is taken by `&mut` because libnrt writes into its tensors;
    /// `inputs` only needs to be readable, so concurrent executions can share an
    /// input set if they want to.
    pub fn execute(&self, inputs: &TensorSet, outputs: &mut TensorSet) -> Result<()> {
        // SAFETY: both sets are live and belong to the same runtime as this
        // model. `nrt_execute` takes the model as `*mut` but is safe to call
        // concurrently on a shared model, which is what the `Sync` impl above
        // relies on.
        unsafe {
            let status =
                (self.runtime.symbols.nrt_execute)(self.ptr, inputs.as_ptr(), outputs.as_ptr());
            self.runtime.symbols.check("nrt_execute", status)
        }
    }

    /// Looks up one tensor's description by name.
    pub fn tensor_info(&self, name: &str, usage: Usage) -> Result<&TensorInfo> {
        self.tensors
            .iter()
            .find(|t| t.name == name && t.usage == usage)
            .ok_or_else(|| Error::NoSuchTensor {
                usage: usage.as_str(),
                name: name.to_owned(),
            })
    }

    /// Reads the NEFF's tensor table and copies it into owned Rust values.
    ///
    /// The C side hands back a heap block with a flexible array member, which
    /// must be released with `nrt_free_model_tensor_info`, so everything is
    /// copied out before that happens.
    fn read_tensor_info(&self) -> Result<Vec<TensorInfo>> {
        let mut array: *mut sys::nrt_tensor_info_array_t = ptr::null_mut();
        // SAFETY: live model, valid out-parameter.
        unsafe {
            let status = (self.runtime.symbols.nrt_get_model_tensor_info)(self.ptr, &mut array);
            self.runtime
                .symbols
                .check("nrt_get_model_tensor_info", status)?;
        }
        if array.is_null() {
            return Ok(Vec::new());
        }

        // SAFETY: `array` is a non-null block libnrt allocated. `tensor_count`
        // is the number of valid entries in the trailing flexible array, so
        // `as_slice` over that count stays in bounds. Freed below in all paths.
        let result = unsafe {
            let count = (*array).tensor_count as usize;
            let entries = (*array).tensor_array.as_slice(count);
            entries
                .iter()
                .map(|entry| {
                    // `shape` is a libnrt-owned array of `ndim` u32s.
                    let shape = if entry.shape.is_null() || entry.ndim == 0 {
                        Vec::new()
                    } else {
                        std::slice::from_raw_parts(entry.shape, entry.ndim as usize).to_vec()
                    };
                    TensorInfo {
                        name: cstr_from_array(&entry.name),
                        usage: Usage::from_raw(entry.usage),
                        size: entry.size,
                        dtype: DataType::from_raw(entry.dtype),
                        shape,
                    }
                })
                .collect::<Vec<_>>()
        };

        // SAFETY: `array` came from `nrt_get_model_tensor_info` and everything
        // borrowed from it has been copied into owned values above.
        unsafe {
            let status = (self.runtime.symbols.nrt_free_model_tensor_info)(array);
            self.runtime
                .symbols
                .check("nrt_free_model_tensor_info", status)?;
        }

        Ok(result)
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("tensors", &self.tensors)
            .finish_non_exhaustive()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `nrt_load` and is unloaded exactly once. The
        // `Arc<Runtime>` guarantees `nrt_close` has not run yet.
        unsafe { (self.runtime.symbols.nrt_unload)(self.ptr) };
    }
}
