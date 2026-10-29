// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! [`InferenceBackend`] over AWS Neuron Runtime, via the workspace's `libnrt-rs` crate.

use std::fmt;
use std::sync::Arc;

use libnrt_rs::{DataType as NrtDataType, Model, Runtime, TensorInfo, TensorSet};

use crate::inference::InferenceError;
use crate::inference::backend::InferenceBackend;
use crate::inference::batch::{
    ElementType, Feature, FeatureData, InputBatch, IoSpec, OutputBatch, ScoreData,
};
use crate::inference::scoring::MiniBatch;
use crate::inference::session_pool::SessionPool;

/// One reusable set of device tensors for an in-flight execution.
#[derive(Debug)]
struct ExecutionContext {
    model_index: usize,
    inputs: TensorSet,
    outputs: TensorSet,
}

/// One logical replica of a compiled NEFF, copied across consecutive NeuronCores.
///
/// Each core holds a complete model copy; this is data parallelism, not model sharding.
/// Device tensors are allocated at startup and leased per call, round-robin across those
/// copies. A request leases the logical replica while its mini batches borrow contexts,
/// so two pieces can execute on two cores without exposing two service-level replicas.
pub(crate) struct NeuronBackend {
    // Contexts must drop before the models whose device tensors they target.
    contexts: SessionPool<ExecutionContext>,
    models: Vec<Model>,
    input_info: Vec<TensorInfo>,
    output_info: Vec<TensorInfo>,
    inputs: Vec<IoSpec>,
    outputs: Vec<IoSpec>,
    batch_rows: usize,
    name: String,
}

impl NeuronBackend {
    /// Loads one NEFF copy per core and allocates reusable execution contexts.
    ///
    /// # Arguments
    ///
    /// * `runtime` - the process-wide Neuron Runtime
    /// * `neff` - compiled Neuron executable bytes
    /// * `start_neuron_core` - first NeuronCore occupied by this logical replica
    /// * `neuron_core_count` - consecutive cores, each holding one complete NEFF copy
    /// * `execution_contexts` - tensor sets distributed across the per-core copies
    /// * `mini_batch` - the fixed row count the NEFF was compiled for
    ///
    /// # Errors
    ///
    /// Fails when a core or context count is invalid, the NEFF cannot load, its signature
    /// is not a supported tabular shape, its fixed batch differs from the configuration,
    /// or device tensors cannot be allocated.
    pub(crate) fn load(
        runtime: Arc<Runtime>,
        neff: &[u8],
        start_neuron_core: i32,
        neuron_core_count: i32,
        execution_contexts: usize,
        mini_batch: Option<MiniBatch>,
    ) -> Result<Self, InferenceError> {
        let core_count = usize::try_from(neuron_core_count)
            .ok()
            .filter(|count| *count > 0)
            .ok_or("a Neuron replica needs at least one NeuronCore")?;
        if execution_contexts < core_count {
            return Err(format!(
                "a Neuron replica using {core_count} core(s) needs at least {core_count} execution context(s), one per core"
            )
            .into());
        }

        let version = runtime.version()?;
        let mut models = Vec::with_capacity(core_count);
        for model_index in 0..core_count {
            let offset = i32::try_from(model_index)
                .map_err(|_| "the NeuronCore offset does not fit libnrt")?;
            let core = start_neuron_core
                .checked_add(offset)
                .ok_or("the requested NeuronCore range is too large")?;
            let model = Model::load(Arc::clone(&runtime), neff, core, 1).map_err(
                |error| -> InferenceError {
                    format!(
                        "could not load Neuron model copy {} of {core_count} on core {core}: {error}",
                        model_index + 1
                    )
                    .into()
                },
            )?;
            models.push(model);
        }

        let primary = models
            .first()
            .expect("a positive core count loaded a model");
        let input_info: Vec<TensorInfo> = primary.inputs().cloned().collect();
        let output_info: Vec<TensorInfo> = primary.outputs().cloned().collect();
        let inputs = convert_specs("input", &input_info, mini_batch)?;
        let outputs = convert_specs("output", &output_info, mini_batch)?;

        if inputs.is_empty() {
            return Err("the NEFF declares no inputs".into());
        }
        if outputs.is_empty() {
            return Err("the NEFF declares no outputs".into());
        }

        let mut allocated = Vec::with_capacity(execution_contexts);
        for index in 0..execution_contexts {
            let model_index = index % core_count;
            let offset = i32::try_from(model_index)
                .map_err(|_| "the NeuronCore offset does not fit libnrt")?;
            let core = start_neuron_core
                .checked_add(offset)
                .ok_or("the requested NeuronCore range is too large")?;
            let (inputs, outputs) = models[model_index].allocate_io(core).map_err(
                |error| -> InferenceError {
                    format!(
                        "could not allocate Neuron execution context {} of {execution_contexts} on core {core}: {error}",
                        index + 1
                    )
                    .into()
                },
            )?;
            allocated.push(ExecutionContext {
                model_index,
                inputs,
                outputs,
            });
        }

        let batch_rows = mini_batch
            .map(|batch| batch.size)
            .ok_or("a NEFF requires a fixed mini batch")?;
        let last_neuron_core = start_neuron_core
            .checked_add(neuron_core_count - 1)
            .ok_or("the requested NeuronCore range is too large")?;
        let placement = if start_neuron_core == last_neuron_core {
            start_neuron_core.to_string()
        } else {
            format!("{start_neuron_core}-{last_neuron_core}")
        };
        let name = format!(
            "neuron/libnrt {version} (NeuronCore {placement}, {core_count} model copy/copies, {execution_contexts} context(s))"
        );

        Ok(Self {
            contexts: SessionPool::new(allocated),
            models,
            input_info,
            output_info,
            inputs,
            outputs,
            batch_rows,
            name,
        })
    }
}

impl fmt::Debug for NeuronBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuronBackend")
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("batch_rows", &self.batch_rows)
            .field("model_copies", &self.models.len())
            .field("execution_contexts", &self.contexts.total())
            .finish_non_exhaustive()
    }
}

impl InferenceBackend for NeuronBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn inputs(&self) -> &[IoSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[IoSpec] {
        &self.outputs
    }

    /// Runs one fixed-size batch on the core associated with the leased context.
    ///
    /// # Errors
    ///
    /// Fails when the batch does not match the NEFF signature, no device tensor exists
    /// for a declared value, execution fails, or an output has the wrong size.
    fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
        let rows = inputs.rows();
        if rows != self.batch_rows {
            return Err(format!(
                "the NEFF was compiled for {} rows, but received {rows}",
                self.batch_rows
            )
            .into());
        }

        let features = inputs.into_features();
        validate_feature_names(&features, &self.inputs)?;
        let mut context = self.contexts.acquire();

        for ((spec, info), feature) in self.inputs.iter().zip(&self.input_info).map(|pair| {
            let feature = features
                .iter()
                .find(|feature| feature.name == pair.0.name)
                .expect("feature names were validated");
            (pair, feature)
        }) {
            validate_feature(spec, feature)?;
            let bytes = feature_bytes(feature, info.dtype)?;
            if bytes.len() != info.size {
                return Err(format!(
                    "input {:?} carries {} bytes, but the NEFF requires {}",
                    spec.name,
                    bytes.len(),
                    info.size
                )
                .into());
            }
            let tensor = context.inputs.get_mut(&info.name).ok_or_else(|| {
                InferenceError::from(format!(
                    "no device tensor was allocated for input {:?}",
                    info.name
                ))
            })?;
            tensor.write_bytes(0, bytes)?;
        }

        let model_index = context.model_index;
        let ExecutionContext {
            inputs, outputs, ..
        } = &mut *context;
        self.models[model_index].execute(inputs, outputs)?;

        let mut batch = OutputBatch::new(rows);
        for ((spec, info), tensor) in self.outputs.iter().zip(&self.output_info).map(|pair| {
            let tensor = outputs.get(&pair.1.name).ok_or_else(|| {
                InferenceError::from(format!(
                    "no device tensor was allocated for output {:?}",
                    pair.1.name
                ))
            });
            (pair, tensor)
        }) {
            let values = tensor?.read_f32()?;
            let expected = element_count(info)?;
            if values.len() != expected {
                return Err(format!(
                    "output {:?} returned {} values, but its NEFF shape {:?} requires {expected}",
                    spec.name,
                    values.len(),
                    info.shape
                )
                .into());
            }
            batch.push(spec.name.clone(), ScoreData::F32(values))?;
        }
        Ok(batch)
    }
}

/// Refuses missing and unexpected inputs before touching device memory.
fn validate_feature_names(features: &[Feature], specs: &[IoSpec]) -> Result<(), InferenceError> {
    for feature in features {
        if !specs.iter().any(|spec| spec.name == feature.name) {
            let wanted: Vec<String> = specs.iter().map(IoSpec::describe).collect();
            return Err(format!(
                "input {:?} is not one this NEFF takes; it takes: {}",
                feature.name,
                wanted.join(", ")
            )
            .into());
        }
    }
    for spec in specs {
        if !features.iter().any(|feature| feature.name == spec.name) {
            return Err(format!(
                "the NEFF requires input {:?}, but it was not supplied",
                spec.name
            )
            .into());
        }
    }
    Ok(())
}

/// Checks one feature against the signature read from the NEFF.
fn validate_feature(spec: &IoSpec, feature: &Feature) -> Result<(), InferenceError> {
    if feature.data.element_type() != spec.element_type {
        return Err(format!(
            "input {:?} is {}, but the NEFF requires {}",
            spec.name,
            feature.data.element_type(),
            spec.element_type
        )
        .into());
    }
    if let Some(width) = spec.width
        && feature.width() != width
    {
        return Err(format!(
            "input {:?} is {} values wide, but the NEFF requires {width}",
            spec.name,
            feature.width()
        )
        .into());
    }
    Ok(())
}

/// Borrows one numeric feature as the native-endian bytes libnrt copies to the device.
fn feature_bytes(feature: &Feature, dtype: NrtDataType) -> Result<&[u8], InferenceError> {
    match (&feature.data, dtype) {
        (FeatureData::F32(values), NrtDataType::Float32 | NrtDataType::Float32R) => {
            Ok(numeric_bytes(values))
        }
        (FeatureData::I32(values), NrtDataType::Int32) => Ok(numeric_bytes(values)),
        (FeatureData::I64(values), NrtDataType::Int64) => Ok(numeric_bytes(values)),
        _ => Err(format!(
            "input {:?} cannot be copied into a {:?} Neuron tensor",
            feature.name, dtype
        )
        .into()),
    }
}

/// Views plain numeric values as bytes without changing their representation.
fn numeric_bytes<T>(values: &[T]) -> &[u8] {
    let size = std::mem::size_of_val(values);
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size) }
}

/// Number of elements declared by one fixed NEFF tensor shape.
fn element_count(info: &TensorInfo) -> Result<usize, InferenceError> {
    info.shape.iter().try_fold(1usize, |count, dimension| {
        let dimension = usize::try_from(*dimension).map_err(|_| {
            InferenceError::from(format!(
                "tensor {:?} has an invalid dimension {dimension}",
                info.name
            ))
        })?;
        count.checked_mul(dimension).ok_or_else(|| {
            InferenceError::from(format!(
                "tensor {:?} shape {:?} is too large",
                info.name, info.shape
            ))
        })
    })
}

/// Translates a fixed NEFF signature into hushar's tabular signature.
fn convert_specs(
    kind: &str,
    specs: &[TensorInfo],
    mini_batch: Option<MiniBatch>,
) -> Result<Vec<IoSpec>, InferenceError> {
    specs
        .iter()
        .map(|info| {
            let element_type = match (kind, info.dtype) {
                (_, NrtDataType::Float32 | NrtDataType::Float32R) => ElementType::F32,
                ("input", NrtDataType::Int32) => ElementType::I32,
                ("input", NrtDataType::Int64) => ElementType::I64,
                ("output", other) => {
                    return Err(format!(
                        "NEFF output {:?} is {:?}, but a response carries FP32 scores",
                        info.name, other
                    )
                    .into());
                }
                (_, other) => {
                    return Err(format!(
                        "NEFF input {:?} is {:?}, but this backend accepts FP32, INT32 or INT64 inputs",
                        info.name, other
                    )
                    .into());
                }
            };

            let rank = info.shape.len();
            if rank == 0 || rank > 2 || info.shape.contains(&0) {
                return Err(format!(
                    "NEFF {kind} {:?} has shape {:?}; tabular tensors must be [batch] or [batch, width] with fixed non-zero dimensions",
                    info.name, info.shape
                )
                .into());
            }

            let pinned = usize::try_from(info.shape[0]).map_err(|_| {
                InferenceError::from(format!(
                    "NEFF {kind} {:?} has an invalid batch dimension {}",
                    info.name, info.shape[0]
                ))
            })?;
            match mini_batch {
                None => {
                    return Err(format!(
                        "NEFF {kind} {:?} fixes its batch at {pinned}; declare \"mini_batch_size\": {pinned} with \"is_fixed\": true",
                        info.name
                    )
                    .into());
                }
                Some(batch) if batch.size != pinned => {
                    return Err(format!(
                        "NEFF {kind} {:?} fixes its batch at {pinned}, but \"mini_batch_size\" is {}",
                        info.name, batch.size
                    )
                    .into());
                }
                Some(batch) if !batch.is_fixed => {
                    return Err(format!(
                        "NEFF {kind} {:?} fixes its batch at {pinned}; set \"is_fixed\": true so a short final batch is padded",
                        info.name
                    )
                    .into());
                }
                Some(_) => {}
            }

            let elements = element_count(info)?;
            let element_size = info.dtype.size_of().ok_or_else(|| {
                InferenceError::from(format!(
                    "NEFF {kind} {:?} has an unknown element size",
                    info.name
                ))
            })?;
            let expected_bytes = elements.checked_mul(element_size).ok_or_else(|| {
                InferenceError::from(format!(
                    "NEFF {kind} {:?} is too large to address",
                    info.name
                ))
            })?;
            if expected_bytes != info.size {
                return Err(format!(
                    "NEFF {kind} {:?} shape {:?} and {:?} imply {expected_bytes} bytes, but libnrt reports {}",
                    info.name, info.shape, info.dtype, info.size
                )
                .into());
            }

            let width = if rank == 1 {
                Some(1)
            } else {
                Some(usize::try_from(info.shape[1]).map_err(|_| {
                    InferenceError::from(format!(
                        "NEFF {kind} {:?} has an invalid width {}",
                        info.name, info.shape[1]
                    ))
                })?)
            };
            Ok(IoSpec::new(info.name.clone(), element_type, width).with_rank(rank))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use libnrt_rs::Usage;

    fn tensor(name: &str, usage: Usage, dtype: NrtDataType, shape: &[u32]) -> TensorInfo {
        let elements: usize = shape.iter().map(|value| *value as usize).product();
        TensorInfo {
            name: name.to_owned(),
            usage,
            size: elements * dtype.size_of().expect("known dtype"),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn fixed(size: usize) -> Option<MiniBatch> {
        Some(MiniBatch {
            size,
            is_fixed: true,
        })
    }

    #[test]
    fn converts_the_supported_neff_signature() {
        let inputs = [
            tensor("features", Usage::Input, NrtDataType::Float32, &[4, 3]),
            tensor("ids", Usage::Input, NrtDataType::Int64, &[4]),
        ];
        let outputs = [tensor(
            "scores",
            Usage::Output,
            NrtDataType::Float32R,
            &[4, 2],
        )];

        let converted_inputs = convert_specs("input", &inputs, fixed(4)).expect("inputs");
        let converted_outputs = convert_specs("output", &outputs, fixed(4)).expect("outputs");

        assert_eq!(
            converted_inputs[0],
            IoSpec::new("features", ElementType::F32, Some(3))
        );
        assert_eq!(
            converted_inputs[1],
            IoSpec::new("ids", ElementType::I64, Some(1)).with_rank(1)
        );
        assert_eq!(
            converted_outputs[0],
            IoSpec::new("scores", ElementType::F32, Some(2))
        );
    }

    #[test]
    fn requires_the_neffs_fixed_batch_configuration() {
        let specs = [tensor(
            "features",
            Usage::Input,
            NrtDataType::Float32,
            &[8, 3],
        )];

        for mini_batch in [
            None,
            Some(MiniBatch {
                size: 4,
                is_fixed: true,
            }),
            Some(MiniBatch {
                size: 8,
                is_fixed: false,
            }),
        ] {
            let error = convert_specs("input", &specs, mini_batch).expect_err("must refuse");
            assert!(error.to_string().contains("batch"), "{error}");
        }
    }

    #[test]
    fn refuses_unservable_types_and_inconsistent_sizes() {
        let bf16 = [tensor(
            "features",
            Usage::Input,
            NrtDataType::BFloat16,
            &[1, 3],
        )];
        assert!(convert_specs("input", &bf16, fixed(1)).is_err());

        let mut bad_size = tensor("scores", Usage::Output, NrtDataType::Float32, &[1, 2]);
        bad_size.size += 4;
        assert!(convert_specs("output", &[bad_size], fixed(1)).is_err());
    }

    #[test]
    fn numeric_features_keep_their_native_representation() {
        let mut batch = InputBatch::new(1);
        batch
            .push("x", FeatureData::I32(vec![1, -2]))
            .expect("valid feature");
        let feature = batch.get("x").expect("x");
        let bytes = feature_bytes(feature, NrtDataType::Int32).expect("supported");
        assert_eq!(bytes.len(), 8);
        assert_eq!(
            i32::from_ne_bytes(bytes[0..4].try_into().expect("four bytes")),
            1
        );
        assert_eq!(
            i32::from_ne_bytes(bytes[4..8].try_into().expect("four bytes")),
            -2
        );
    }
}
