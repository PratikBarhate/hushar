// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! [`InferenceBackend`] over ONNX Runtime, via the workspace's `onnxrt-rs` crate.
//!
//! The only file in the service that knows ONNX Runtime exists, and the only one that
//! translates between hushar's [`ElementType`] and the engine's.

use std::fmt;
use std::sync::Arc;

use onnxrt_rs::{
    Api, DataType as OrtDataType, Environment, ExecutionMode, ExecutionProvider,
    GraphOptimizationLevel, InputValue, OwnedTensor, Session, SessionBuilder,
};

use crate::config::Parallelism;
use crate::inference::InferenceError;
use crate::inference::backend::InferenceBackend;
use crate::inference::batch::{
    ElementType, Feature, FeatureData, InputBatch, IoSpec, OutputBatch, ScoreData,
};

/// ONNX Runtime, running on whichever execution provider was requested.
///
/// The whole signature is read from the model rather than configured, so a configuration
/// that disagrees with it is reported rather than silently reshaped.
///
/// ONNX Runtime requires the environment to outlive every session created in it. Field
/// order is drop order, so `session` is declared first and released first, and the `Arc`
/// keeps the environment alive for as long as the session needs it.
///
/// The environment is shared rather than per-backend: ONNX Runtime allows only one
/// default-logger environment per process, so creating one per backend would fail
/// whenever two models are loaded at once.
pub(crate) struct OnnxRuntimeBackend {
    session: Session,
    _environment: Arc<Environment>,
    api: Api,
    inputs: Vec<IoSpec>,
    outputs: Vec<IoSpec>,
    name: String,
}

impl OnnxRuntimeBackend {
    /// Loads an ONNX model and prepares it to run on `provider`.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - the ONNX model protobuf
    /// * `provider` - the execution provider the session runs on
    /// * `parallelism` - which of ONNX Runtime's thread pools carries this session's width,
    ///   and how wide. Sets the execution mode to match, since inter-op threads are built
    ///   and never given work under the sequential executor. A width of `0` leaves that
    ///   pool at the runtime's own default.
    /// * `mini_batch` - how the deployment cuts a request's rows, when the model
    ///   configuration says to. A padded batch is what allows a model with a pinned
    ///   leading dimension to load at all; see [`convert_specs`].
    /// * `session_options` - ONNX Runtime's string-keyed session settings, applied after
    ///   the thread count because an affinity list is validated against it. See
    ///   [`onnxrt_rs::SessionBuilder::config_entry`].
    ///
    /// # Errors
    ///
    /// Fails if the ONNX Runtime library cannot be loaded, if `provider` is not compiled
    /// into it (the error names the providers that are), if the model will not parse, or
    /// if the model uses an element type this service does not serve.
    pub(crate) fn load(
        model_bytes: &[u8],
        provider: &ExecutionProvider,
        parallelism: Parallelism,
        mini_batch: Option<crate::inference::scoring::MiniBatch>,
        session_options: &[(String, String)],
    ) -> Result<Self, InferenceError> {
        let api = Api::load()?;
        let environment = Environment::shared("hushar")?;

        let mut builder = SessionBuilder::new(environment.as_ref())?
            .optimization_level(GraphOptimizationLevel::All)?
            .execution_provider(provider)?
            .execution_mode(if parallelism.is_parallel() {
                ExecutionMode::Parallel
            } else {
                ExecutionMode::Sequential
            })?;
        match parallelism {
            Parallelism::Intra(threads) => {
                if threads > 0 {
                    builder = builder.intra_op_threads(threads)?;
                }
            }
            Parallelism::Inter(threads) => {
                // The intra-op pool does not go away when the width moves to inter-op, and
                // its default is one thread per physical core. Left unset, this session
                // would hold the machine once for that pool and again for this one, so it
                // is pinned to the single calling thread. The configuration has already
                // refused any intra-op count other than 1 here.
                builder = builder.intra_op_threads(1)?;
                if threads > 0 {
                    builder = builder.inter_op_threads(threads)?;
                }
            }
        }
        for (key, value) in session_options {
            builder = builder.config_entry(key, value)?;
        }
        let session = builder.build_from_memory(model_bytes)?;

        let inputs = convert_specs("input", &session.input_specs()?, mini_batch)?;
        let outputs = convert_specs("output", &session.output_specs()?, mini_batch)?;

        let name = format!("onnxruntime/{provider} (runtime {})", api.version());

        Ok(Self {
            session,
            _environment: environment,
            api,
            inputs,
            outputs,
            name,
        })
    }
}

impl fmt::Debug for OnnxRuntimeBackend {
    /// Hand-written because `Session` and `Environment` wrap raw C handles, which have no
    /// useful representation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxRuntimeBackend")
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}

impl InferenceBackend for OnnxRuntimeBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn inputs(&self) -> &[IoSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[IoSpec] {
        &self.outputs
    }

    /// Runs one batch.
    ///
    /// Features are taken by value so their buffers can be lent to the engine mutably:
    /// `CreateTensorWithDataAsOrtValue` wraps this memory rather than copying it. Each
    /// input's shape comes from the model's own spec rather than from `[rows, width]`,
    /// because the engine checks rank: an input declared `[batch]` will not accept a
    /// two-dimensional shape even when the values are the same.
    ///
    /// Outputs come back in declared order and are named by zipping, so their count, row
    /// count and element types are checked against the signature.
    ///
    /// # Errors
    ///
    /// Fails if an input is not one the model takes, if the engine rejects the batch, or
    /// if the outputs disagree with the declared signature in count, rows or type.
    fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
        let rows = inputs.rows();
        let mut owned: Vec<Feature> = inputs.into_features();

        let names: Vec<String> = owned.iter().map(|f| f.name.clone()).collect();
        let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(owned.len());
        for feature in &owned {
            let spec = self
                .inputs
                .iter()
                .find(|s| s.name == feature.name)
                .ok_or_else(|| {
                    let wanted: Vec<String> = self.inputs.iter().map(IoSpec::describe).collect();
                    format!(
                        "input {:?} is not one this model takes; it takes: {}",
                        feature.name,
                        wanted.join(", ")
                    )
                })?;
            shapes.push(spec.engine_shape(rows, feature.width()));
        }

        let mut values: Vec<InputValue<'_>> = Vec::with_capacity(owned.len());
        for ((feature, shape), name) in owned.iter_mut().zip(shapes.iter()).zip(names.iter()) {
            let value = to_input_value(self.api, &mut feature.data, shape)
                .map_err(|e| -> InferenceError { format!("input {name:?}: {e}").into() })?;
            values.push(value);
        }

        let by_name: Vec<(&str, InputValue<'_>)> =
            names.iter().map(|n| n.as_str()).zip(values).collect();

        let produced = self.session.run_named(&by_name)?;

        if produced.len() != self.outputs.len() {
            return Err(format!(
                "model returned {} outputs but declares {}",
                produced.len(),
                self.outputs.len()
            )
            .into());
        }

        let mut out = OutputBatch::new(rows);
        for (spec, tensor) in self.outputs.iter().zip(produced.iter()) {
            let (data, produced_rows) = from_owned_tensor(&spec.name, tensor)?;
            if produced_rows != rows {
                return Err(format!(
                    "model output {:?} has {produced_rows} rows for a batch of {rows}",
                    spec.name
                )
                .into());
            }
            if data.element_type() != spec.element_type {
                return Err(format!(
                    "output {:?} was declared {} but came back as {}",
                    spec.name,
                    spec.element_type,
                    data.element_type()
                )
                .into());
            }
            out.push(spec.name.clone(), data)?;
        }
        Ok(out)
    }
}

/// Wraps a feature's data as an ONNX Runtime input.
///
/// Numeric types borrow the buffer. Text has to copy: the ONNX Runtime C API has no
/// borrow-based equivalent of `FillStringTensor`.
fn to_input_value<'d>(
    api: Api,
    data: &'d mut FeatureData,
    shape: &[i64],
) -> Result<InputValue<'d>, InferenceError> {
    Ok(match data {
        FeatureData::F32(v) => InputValue::numeric(api, v, shape)?,
        FeatureData::F64(v) => InputValue::numeric(api, v, shape)?,
        FeatureData::I32(v) => InputValue::numeric(api, v, shape)?,
        FeatureData::I64(v) => InputValue::numeric(api, v, shape)?,
        FeatureData::Str(v) => InputValue::strings(api, v, shape)?,
    })
}

/// Reads an engine output back into hushar's representation, with its row count.
///
/// # Errors
///
/// Fails if the leading dimension is not a row count, or if the tensor is neither `FP32`
/// nor `FP64` -- which [`convert_specs`] refuses at load, so it means the model's
/// declared and actual output types disagree.
fn from_owned_tensor(
    name: &str,
    tensor: &OwnedTensor,
) -> Result<(ScoreData, usize), InferenceError> {
    let shape = tensor.shape()?;
    let rows = usize::try_from(shape.first().copied().unwrap_or(0)).map_err(|_| {
        format!(
            "model output {name:?} came back with shape {shape:?}, whose leading \
             dimension is not a row count"
        )
    })?;
    let data = match tensor.data_type()? {
        OrtDataType::F32 => ScoreData::F32(tensor.to_vec::<f32>()?),
        OrtDataType::F64 => ScoreData::F64(tensor.to_vec::<f64>()?),
        other => {
            return Err(format!(
                "model output {name:?} came back as {}, which this service does not \
                 serve; the signature declared something else than it produced",
                other.name()
            )
            .into());
        }
    };
    Ok((data, rows))
}

/// Translates the engine's declared signature into hushar's, refusing what it cannot
/// serve.
///
/// Runs when the model loads, so an unservable model is a startup failure rather than a
/// request that fails once traffic arrives. A `[batch]` axis carries one value per row; a
/// dynamic width is left unknown, for the configuration to decide. `mini_batch` is
/// checked per tensor rather than across the signature, because a graph can pin some axes
/// and leave others symbolic.
///
/// # Arguments
///
/// * `kind` - `"input"` or `"output"`; decides which element types are allowed and how
///   the refusal reads
/// * `specs` - the tensors the engine declares
/// * `mini_batch` - the padded batch size the deployment declares, if any
///
/// # Errors
///
/// Fails, naming the tensor and its type, when:
///
/// * the element type is unservable: an input may be `FP32`, `FP64`, `INT32`, `INT64` or
///   `STRING`, an output only `FP32` or `FP64`, because a response carries scores
/// * the rank is not `[batch]` or `[batch, width]`
/// * the leading axis is pinned and `mini_batch` does not declare a matching size with
///   `is_fixed` set, so every batch would be padded up to it
fn convert_specs(
    kind: &str,
    specs: &[onnxrt_rs::TensorSpec],
    mini_batch: Option<crate::inference::scoring::MiniBatch>,
) -> Result<Vec<IoSpec>, InferenceError> {
    specs
        .iter()
        .map(|s| {
            let element_type = match (kind, s.data_type) {
                (_, OrtDataType::F32) => ElementType::F32,
                (_, OrtDataType::F64) => ElementType::F64,
                ("input", OrtDataType::I32) => ElementType::I32,
                ("input", OrtDataType::I64) => ElementType::I64,
                ("input", OrtDataType::String) => ElementType::Str,
                ("output", other) => {
                    return Err(InferenceError::from(format!(
                        "model output {:?} is {}, but a response carries FP32 or FP64 \
                         scores. Export the model with a float output, or add a \
                         post-processing step that produces one.",
                        s.name,
                        other.name()
                    )));
                }
                (_, other) => {
                    return Err(InferenceError::from(format!(
                        "model input {:?} is {}, but an input may be FP32, FP64, INT32, \
                         INT64 or STRING. FP16 and BF16 have no protobuf field, and the \
                         narrower integers share protobuf's int32 field, so they would \
                         need a range check per element.",
                        s.name,
                        other.name()
                    )));
                }
            };

            let rank = s.shape.len();
            if rank == 0 || rank > 2 {
                return Err(InferenceError::from(format!(
                    "model {kind} {:?} has shape {:?}. This service serves tabular \
                     models, whose inputs and outputs are [batch] or [batch, width]; \
                     flatten the extra axes into the width, or serve this model \
                     elsewhere.",
                    s.name, s.shape
                )));
            }
            if s.shape[0] >= 0 {
                let pinned = usize::try_from(s.shape[0]).unwrap_or(usize::MAX);
                match mini_batch {
                    None => {
                        return Err(InferenceError::from(format!(
                            "model {kind} {:?} fixes its leading dimension at {}, so it \
                             could only ever be given batches of that size. Re-export \
                             the model with a dynamic batch axis, or declare \
                             \"mini_batch_size\": {} with \"is_fixed\": true in the \
                             model configuration to pad every batch up to that size.",
                            s.name, s.shape[0], s.shape[0]
                        )));
                    }
                    Some(m) if m.size != pinned => {
                        return Err(InferenceError::from(format!(
                            "model {kind} {:?} fixes its leading dimension at {}, but \
                             the model configuration declares \"mini_batch_size\": {}. \
                             Every request would fail inside the engine on a shape \
                             mismatch. Set mini_batch_size to {} or re-export the model \
                             for {} rows.",
                            s.name, s.shape[0], m.size, s.shape[0], m.size
                        )));
                    }
                    Some(m) if !m.is_fixed => {
                        return Err(InferenceError::from(format!(
                            "model {kind} {:?} fixes its leading dimension at {}, and \
                             the model configuration declares \"mini_batch_size\": {} \
                             without \"is_fixed\": true. A request whose row count is \
                             not a multiple of {} would send a short last batch and fail \
                             inside the engine. Add \"is_fixed\": true so the last batch \
                             is padded up to {}.",
                            s.name, s.shape[0], m.size, m.size, m.size
                        )));
                    }
                    Some(_) => {}
                }
            }

            let width = if rank == 1 {
                Some(1)
            } else {
                usize::try_from(s.shape[1]).ok()
            };

            Ok(IoSpec::new(s.name.clone(), element_type, width).with_rank(rank))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, data_type: OrtDataType) -> onnxrt_rs::TensorSpec {
        shaped(name, data_type, vec![-1, 3])
    }

    fn shaped(name: &str, data_type: OrtDataType, shape: Vec<i64>) -> onnxrt_rs::TensorSpec {
        onnxrt_rs::TensorSpec {
            name: name.to_owned(),
            data_type,
            shape,
        }
    }

    #[test]
    fn the_five_served_types_pass_the_signature_gate() {
        let specs = [
            spec("a", OrtDataType::F32),
            spec("b", OrtDataType::F64),
            spec("c", OrtDataType::I32),
            spec("d", OrtDataType::I64),
            spec("e", OrtDataType::String),
        ];
        let converted = convert_specs("input", &specs, None).expect("all five are servable");
        assert_eq!(
            converted.iter().map(|s| s.element_type).collect::<Vec<_>>(),
            vec![
                ElementType::F32,
                ElementType::F64,
                ElementType::I32,
                ElementType::I64,
                ElementType::Str
            ]
        );
    }

    /// An `INT32` input is served, an `INT32` output refused.
    #[test]
    fn an_integer_input_is_served_but_an_integer_output_is_not() {
        assert!(
            convert_specs("input", &[spec("code", OrtDataType::I32)], None).is_ok(),
            "INT32 has a protobuf field of its own, so it costs nothing to serve"
        );
        let err = convert_specs("output", &[spec("code", OrtDataType::I32)], None)
            .expect_err("a response carries FP32 or FP64 scores");
        assert!(err.to_string().contains("code"), "got {err}");
    }

    /// An unservable element type is refused, naming the tensor, the type found and the
    /// types that are served.
    #[test]
    fn a_non_float_signature_is_refused_with_the_tensor_named() {
        for ort in [
            OrtDataType::F16,
            OrtDataType::Bf16,
            OrtDataType::Bool,
            OrtDataType::I8,
            OrtDataType::I16,
            OrtDataType::U8,
            OrtDataType::U16,
            OrtDataType::U32,
            OrtDataType::U64,
        ] {
            let err = convert_specs("input", &[spec("tags", ort)], None)
                .expect_err("{ort} is not servable");
            let message = err.to_string();
            assert!(
                message.contains("tags"),
                "{ort}: should name the tensor: {message}"
            );
            assert!(
                message.contains(ort.name()),
                "{ort}: should name the type it found: {message}"
            );
            assert!(
                message.contains("FP32")
                    && message.contains("FP64")
                    && message.contains("INT32")
                    && message.contains("INT64")
                    && message.contains("STRING"),
                "{ort}: should name what is served: {message}"
            );
        }
    }

    /// The refusal says "model input" or "model output" to match the side it came from.
    #[test]
    fn the_refusal_says_input_or_output_depending_on_which_it_is() {
        let inputs = convert_specs("input", &[spec("x", OrtDataType::I16)], None)
            .expect_err("not servable")
            .to_string();
        assert!(inputs.contains("model input"), "got {inputs}");

        let outputs = convert_specs("output", &[spec("x", OrtDataType::I16)], None)
            .expect_err("not servable")
            .to_string();
        assert!(outputs.contains("model output"), "got {outputs}");
    }

    #[test]
    fn specs_carry_names_types_and_widths_across() {
        let engine = [
            spec("numbers", OrtDataType::F32),
            shaped("totals", OrtDataType::F64, vec![-1, 1]),
        ];
        let ours = convert_specs("input", &engine, None).expect("conversion");
        assert_eq!(ours.len(), 2);
        assert_eq!(ours[0].name, "numbers");
        assert_eq!(ours[0].element_type, ElementType::F32);
        assert_eq!(ours[0].width, Some(3));
        assert_eq!(ours[0].rank, 2);
        assert_eq!(ours[1].element_type, ElementType::F64);
        assert_eq!(ours[1].width, Some(1));
    }

    /// A rank-1 `[batch]` tensor reads as width 1 and keeps its rank.
    #[test]
    fn a_batch_axis_carries_one_value_per_row() {
        let ours = convert_specs(
            "input",
            &[shaped("tags", OrtDataType::String, vec![-1])],
            None,
        )
        .expect("rank 1 is servable");
        assert_eq!(ours[0].width, Some(1));
        assert_eq!(ours[0].rank, 1);
        assert_eq!(ours[0].engine_shape(4, 1), vec![4]);
    }

    #[test]
    fn a_dynamic_width_reads_as_unknown_rather_than_as_a_number() {
        let ours = convert_specs(
            "input",
            &[shaped("x", OrtDataType::F32, vec![-1, -1])],
            None,
        )
        .expect("a dynamic width is servable");
        assert_eq!(
            ours[0].width, None,
            "the configuration decides what the model leaves open"
        );
    }

    /// A rank above two is refused, naming the tensor and the shapes that are served.
    #[test]
    fn a_rank_above_two_is_refused_when_the_model_loads() {
        let err = convert_specs(
            "input",
            &[shaped("images", OrtDataType::F32, vec![-1, 3, 32])],
            None,
        )
        .expect_err("rank 3 is not tabular");
        let m = err.to_string();
        assert!(m.contains("images"), "should name the tensor: {m}");
        assert!(
            m.contains("[batch, width]"),
            "should say what is served: {m}"
        );
    }

    #[test]
    fn a_scalar_signature_is_refused() {
        let err = convert_specs("output", &[shaped("total", OrtDataType::F32, vec![])], None)
            .expect_err("a rank-0 output has no rows");
        assert!(err.to_string().contains("total"), "got {err}");
    }

    /// An undeclared pinned batch dimension is refused, with both fixes named.
    #[test]
    fn a_fixed_batch_dimension_is_refused_with_the_fix_named() {
        let err = convert_specs("input", &[shaped("x", OrtDataType::F32, vec![8, 3])], None)
            .expect_err("a fixed batch cannot be served undeclared");
        let m = err.to_string();
        assert!(m.contains('8'), "should name the dimension found: {m}");
        assert!(m.contains("dynamic batch"), "should name the fix: {m}");
        assert!(
            m.contains("mini_batch_size"),
            "should name the other fix: {m}"
        );
    }

    /// A pinned axis the configuration declares is servable, and the width survives the
    /// check unchanged.
    #[test]
    fn a_declared_batch_size_makes_a_pinned_model_servable() {
        let ours = convert_specs(
            "input",
            &[shaped("x", OrtDataType::F32, vec![1, 502])],
            Some(crate::inference::scoring::MiniBatch {
                size: 1,
                is_fixed: true,
            }),
        )
        .expect("a pinned axis the config declares is servable");
        assert_eq!(ours[0].width, Some(502));
        assert_eq!(ours[0].rank, 2);
    }

    /// A declared size that disagrees with the model's pinned axis is refused, with both
    /// numbers in the message.
    #[test]
    fn a_declared_batch_size_that_disagrees_with_the_model_is_refused() {
        let err = convert_specs(
            "input",
            &[shaped("x", OrtDataType::F32, vec![8, 3])],
            Some(crate::inference::scoring::MiniBatch {
                size: 1,
                is_fixed: true,
            }),
        )
        .expect_err("8 rows cannot serve a deployment promising 1");
        let m = err.to_string();
        assert!(m.contains('8'), "should name the model's dimension: {m}");
        assert!(m.contains('1'), "should name the declared size: {m}");
    }

    /// A pinned axis declared at the right size but without `is_fixed` is refused, naming
    /// the flag to add.
    #[test]
    fn a_pinned_axis_without_the_fixed_flag_is_refused() {
        let err = convert_specs(
            "input",
            &[shaped("x", OrtDataType::F32, vec![8, 3])],
            Some(crate::inference::scoring::MiniBatch {
                size: 8,
                is_fixed: false,
            }),
        )
        .expect_err("a pinned axis needs every batch padded up to it");
        let m = err.to_string();
        assert!(m.contains("is_fixed"), "should name the flag to add: {m}");
    }

    /// A declared mini batch leaves a dynamic leading axis servable, unpadded.
    #[test]
    fn a_declared_batch_size_leaves_a_dynamic_axis_alone() {
        let ours = convert_specs(
            "input",
            &[shaped("x", OrtDataType::F32, vec![-1, 502])],
            Some(crate::inference::scoring::MiniBatch {
                size: 4,
                is_fixed: false,
            }),
        )
        .expect("a dynamic axis is servable whatever the declaration");
        assert_eq!(ours[0].engine_shape(4, 502), vec![4, 502]);
    }

    /// Loads one of the typed fixtures, or explains the skip.
    fn fixture(name: &str) -> Option<Vec<u8>> {
        let path = std::path::Path::new("test-data").join(name);
        match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                eprintln!(
                    "skipping: cannot read {} ({e}). Regenerate with \
                     `python scripts/generate_typed_test_models.py`",
                    path.display()
                );
                None
            }
        }
    }

    /// A CPU backend over `name`, or `None` when the runtime is unavailable.
    fn cpu_backend(name: &str) -> Option<OnnxRuntimeBackend> {
        let model = fixture(name)?;
        match OnnxRuntimeBackend::load(
            &model,
            &ExecutionProvider::Cpu,
            Parallelism::Intra(1),
            None,
            &[],
        ) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping: no ONNX Runtime available ({e})");
                None
            }
        }
    }

    /// A batch of `rows` rows carrying the given features.
    fn batch(rows: usize, features: Vec<(&str, FeatureData)>) -> InputBatch {
        let mut b = InputBatch::new(rows);
        for (name, data) in features {
            b.push(name, data).expect("valid feature");
        }
        b
    }

    /// A model with float and text inputs loads and reports both element types.
    #[test]
    fn a_float_and_text_model_loads_and_reports_both_types() {
        let Some(backend) = cpu_backend("numbers_and_text.onnx") else {
            return;
        };
        let inputs: Vec<(&str, ElementType)> = backend
            .inputs()
            .iter()
            .map(|s| (s.name.as_str(), s.element_type))
            .collect();
        assert_eq!(
            inputs,
            vec![("numbers", ElementType::F32), ("tags", ElementType::Str)]
        );

        let outputs: Vec<(&str, ElementType)> = backend
            .outputs()
            .iter()
            .map(|s| (s.name.as_str(), s.element_type))
            .collect();
        assert_eq!(
            outputs,
            vec![("scores", ElementType::F32), ("code", ElementType::F32)]
        );
    }

    /// A model whose output is `INT64` is refused at load, naming the output and saying
    /// it is an output problem.
    ///
    /// A runtime that is absent, unloadable, or too old for the checked-in bindings is a
    /// reason to skip rather than to fail.
    #[test]
    fn a_model_with_a_non_float_output_is_refused_when_it_loads() {
        let Some(bytes) = fixture("mixed_io.onnx") else {
            return;
        };
        let err = match OnnxRuntimeBackend::load(
            &bytes,
            &ExecutionProvider::Cpu,
            Parallelism::Intra(1),
            None,
            &[],
        ) {
            Ok(_) => panic!("an INT64 output must not be servable"),
            Err(e) => e.to_string(),
        };
        if err.to_lowercase().contains("dylib")
            || err.contains("no ONNX Runtime")
            || err.contains("API version")
        {
            eprintln!("skipping: no usable ONNX Runtime available ({err})");
            return;
        }
        assert!(err.contains("codes"), "should name the output: {err}");
        assert!(
            err.contains("FP32") && err.contains("FP64"),
            "should name what a response carries: {err}"
        );
        assert!(err.contains("output"), "got {err}");
    }

    /// One call to a real model returns logits and an embedding of different widths, each
    /// keeping its own width and values.
    #[test]
    fn a_models_several_outputs_keep_their_own_widths() {
        let Some(backend) = cpu_backend("float_multi_io.onnx") else {
            return;
        };

        let out = backend
            .run(batch(
                2,
                vec![(
                    "numbers",
                    FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                )],
            ))
            .expect("inference should succeed");

        let described: Vec<(&str, ElementType, usize)> = out
            .iter()
            .map(|c| (c.name.as_str(), c.data.element_type(), c.width()))
            .collect();
        assert_eq!(
            described,
            vec![
                ("scores", ElementType::F32, 2),
                ("embedding", ElementType::F32, 4),
            ]
        );
        assert_eq!(out.rows(), 2, "every output shares the batch's row count");

        let scores = out
            .get("scores")
            .expect("scores")
            .row_as_f32(0)
            .expect("row");
        assert!(
            (scores[0] - 0.9568927).abs() < 1e-5,
            "scores were {scores:?}"
        );

        let embedding = out
            .get("embedding")
            .expect("embedding")
            .row_as_f32(0)
            .expect("row");
        for (got, want) in embedding.iter().zip([3.2, 3.8, 4.4, 5.0]) {
            assert!((got - want).abs() < 1e-5, "embedding was {embedding:?}");
        }
    }

    /// An `FP64` model runs end to end against a real engine. The fixture computes
    /// `2x + 1`, exactly representable for these values.
    #[test]
    fn a_double_precision_model_runs_end_to_end() {
        let Some(backend) = cpu_backend("f64_scale.onnx") else {
            return;
        };
        assert_eq!(backend.inputs()[0].element_type, ElementType::F64);
        assert_eq!(backend.outputs()[0].element_type, ElementType::F64);

        let out = backend
            .run(batch(
                1,
                vec![("input", FeatureData::F64(vec![1.5, 2.5, 3.5]))],
            ))
            .expect("fp64 inference should succeed");
        let first = out.first().expect("one output");
        assert_eq!(first.data.element_type(), ElementType::F64);
        assert_eq!(first.data, ScoreData::F64(vec![4.0, 6.0, 8.0]));
    }

    #[test]
    fn the_backend_reads_its_signature_from_the_model() {
        let Some(backend) = cpu_backend("float_multi_io.onnx") else {
            return;
        };

        let inputs: Vec<(&str, ElementType)> = backend
            .inputs()
            .iter()
            .map(|s| (s.name.as_str(), s.element_type))
            .collect();
        assert_eq!(inputs, vec![("numbers", ElementType::F32)]);
        assert_eq!(backend.inputs()[0].width, Some(3));
        assert_eq!(backend.inputs()[0].rank, 2);

        let outputs: Vec<&str> = backend.outputs().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(outputs, vec!["scores", "embedding"]);
    }

    /// A misnamed input is rejected even when the model declares a single input.
    #[test]
    fn a_misnamed_input_is_rejected_by_name() {
        let Some(backend) = cpu_backend("float_multi_io.onnx") else {
            return;
        };

        let err = backend
            .run(batch(
                1,
                vec![("number", FeatureData::F32(vec![1.0, 2.0, 3.0]))],
            ))
            .expect_err("a misnamed input must fail");
        let message = err.to_string();
        assert!(
            message.contains("numbers"),
            "the error should name the input it wanted: {message}"
        );
    }

    /// A vectorised configuration scores through a real model: one concatenated `FP32`
    /// input, as a tabular scorer takes. The fixture computes
    /// `sigmoid(sum(x) * 0.5 + 0.1)`.
    #[test]
    fn a_vectorised_configuration_scores_through_a_real_model() {
        let Some(backend) = cpu_backend("float_multi_io.onnx") else {
            return;
        };

        let model_config = crate::config::ModelConfig::from_json(
            r#"{
                "model_id": "vectorised",
                "model_path": "test-data/float_multi_io.onnx",
                "vectorization_config": {
                    "data_type": "float",
                    "feature_transformations": {
                        "a": {"type": "identity", "default_val": [0.0]},
                        "b": {"type": "identity", "default_val": [0.0]},
                        "c": {"type": "identity", "default_val": [0.0]}
                    },
                    "feature_order": ["a", "b", "c"]
                }
            }"#,
        )
        .expect("a valid model configuration");

        let builder = crate::inference::input_builder::InputBuilder::resolve(
            model_config.vectorization_config,
            backend.inputs(),
        )
        .expect("three features fill the model's three-wide input");

        use crate::inference::input_builder::test_support::{V, request, row};
        let columns = request(&[row(
            "r1",
            &[("a", V::f32(1.0)), ("b", V::f32(2.0)), ("c", V::f32(3.0))],
        )]);

        let scored = crate::inference::scoring::score_features(
            &backend,
            &columns,
            &builder,
            None,
            crate::inference::scoring::FeatureLogging::Record,
        )
        .expect("the batch should score");

        let scores = match &scored.scores {
            ScoreData::F32(v) => v.clone(),
            other => panic!("expected float scores, got {other:?}"),
        };
        assert!((scores[0] - 0.9568927).abs() < 1e-5, "got {scores:?}");
    }

    /// A request of any row count is served through a graph whose leading dimension is
    /// pinned at 2, which ONNX Runtime otherwise refuses outright.
    ///
    /// Covers 1, 3 and 4 rows: under the pinned size, ragged, and an exact multiple. The
    /// fixture scores `sigmoid(0.5 * sum(row) + 0.1)`, so a row of ones gives ~0.832 and
    /// an all-zero padding row ~0.525. Scores come back as one row-major buffer, so a
    /// padded row leaking into the response shows up in the length and in the values.
    #[test]
    fn a_pinned_graph_serves_a_request_of_any_row_count() {
        let Some(model) = fixture("sigmoid_model_3_batch2.onnx") else {
            return;
        };
        let backend = match OnnxRuntimeBackend::load(
            &model,
            &ExecutionProvider::Cpu,
            Parallelism::Intra(1),
            Some(crate::inference::scoring::MiniBatch {
                size: 2,
                is_fixed: true,
            }),
            &[],
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: no ONNX Runtime available ({e})");
                return;
            }
        };

        let config = crate::config::vectorization_config::VectorizationConfig::from_json(
            r#"{"data_type": "float", "feature_order": ["a", "b", "c"]}"#,
        )
        .expect("a valid configuration");
        let builder =
            crate::inference::input_builder::InputBuilder::resolve(config, backend.inputs())
                .expect("three features fill the model's one 3-wide input");

        use crate::inference::input_builder::test_support::{V, request, row};

        let width = backend.outputs()[0]
            .width
            .expect("the pinned fixture declares a fixed output width");

        const REAL: f32 = 0.832_018_4;
        const PAD: f32 = 0.524_979_2;

        for count in [1usize, 3, 4] {
            let ids: Vec<String> = (0..count).map(|i| format!("r{i}")).collect();
            let rows: Vec<_> = ids
                .iter()
                .map(|id| {
                    row(
                        id.as_str(),
                        &[("a", V::f32(1.0)), ("b", V::f32(1.0)), ("c", V::f32(1.0))],
                    )
                })
                .collect();
            let columns = request(&rows);
            let scored = crate::inference::scoring::score_features(
                &backend,
                &columns,
                &builder,
                Some(crate::inference::scoring::MiniBatch {
                    size: 2,
                    is_fixed: true,
                }),
                crate::inference::scoring::FeatureLogging::Record,
            )
            .unwrap_or_else(|e| panic!("{count} rows against a pinned graph: {e}"));

            let scores = match &scored.scores {
                ScoreData::F32(v) => v.clone(),
                other => panic!("expected float scores, got {other:?}"),
            };
            assert_eq!(
                scores.len(),
                count * width,
                "{count} rows in must be {count} rows out"
            );

            for i in 0..count {
                let s = &scores[i * width..(i + 1) * width];
                assert_eq!(
                    columns.row_ids()[i],
                    format!("r{i}"),
                    "row order must survive"
                );
                assert!(
                    (s[0] - REAL).abs() < 1e-5,
                    "row {i} of {count} scored {s:?}, expected a real row's {REAL}"
                );
                assert!(
                    (s[0] - PAD).abs() > 1e-3,
                    "row {i} of {count} scored {s:?}, which is a padding row's score"
                );
            }
        }
    }
}
