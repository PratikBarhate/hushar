// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! [`InferenceBackend`] over ONNX Runtime, via the workspace's `onnxrt-rs` crate.
//!
//! This is the only file in the service that knows ONNX Runtime exists, and the only
//! one that translates between hushar's [`ElementType`] and the engine's. That
//! translation is the price of keeping the layer above engine-agnostic.

use std::fmt;
use std::sync::Arc;

use onnxrt_rs::{
    Api, DataType as OrtDataType, Environment, ExecutionProvider, GraphOptimizationLevel,
    InputValue, OwnedTensor, Session, SessionBuilder,
};

use crate::inference::InferenceError;
use crate::inference::backend::InferenceBackend;
use crate::inference::batch::{
    ElementType, Feature, FeatureData, InputBatch, IoSpec, OutputBatch, ScoreData,
};

/// ONNX Runtime, running on whichever execution provider was requested.
///
/// The whole signature is read from the model rather than configured, so a
/// configuration that disagrees with it is reported rather than silently reshaped.
///
/// # Field order is drop order
///
/// ONNX Runtime requires the environment to outlive every session created in it, so
/// `session` is declared first and is therefore released first. Holding the `Arc`
/// makes that guarantee structural rather than a convention this file remembers.
///
/// The environment is shared rather than per-backend because ONNX Runtime allows only
/// one default-logger environment per process: creating one per backend would fail
/// whenever two models are loaded at once, such as reloading a model while still
/// serving from the old one.
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
    /// `session_options` are ONNX Runtime's string-keyed session settings, applied after
    /// the thread count because an affinity list is validated against it. See
    /// [`onnxrt_rs::SessionBuilder::config_entry`].
    ///
    /// `intra_op_threads` bounds the threads ONNX Runtime uses *inside* one
    /// operator. Pass `Some(1)` when the caller already keeps every core busy
    /// with concurrent requests, as this service does: extra intra-op threads
    /// then only add contention. `None` leaves ONNX Runtime's default.
    ///
    /// `mini_batch` is how the deployment cuts a request's rows, when the
    /// configuration says to. Padded batches are what allow a model with a pinned
    /// leading dimension to load at all; see [`convert_specs`].
    ///
    /// # Errors
    ///
    /// Fails if the ONNX Runtime library cannot be loaded, if `provider` is not
    /// compiled into it (the error names the providers that are), if the model
    /// will not parse, or if the model uses an element type this service does not
    /// model.
    pub(crate) fn load(
        model_bytes: &[u8],
        provider: &ExecutionProvider,
        intra_op_threads: Option<i32>,
        mini_batch: Option<crate::inference::scoring::MiniBatch>,
        session_options: &[(String, String)],
    ) -> Result<Self, InferenceError> {
        let api = Api::load()?;
        let environment = Environment::shared("hushar")?;

        let mut builder = SessionBuilder::new(environment.as_ref())?
            .optimization_level(GraphOptimizationLevel::All)?
            .execution_provider(provider)?;
        if let Some(threads) = intra_op_threads {
            builder = builder.intra_op_threads(threads)?;
        }
        // After the thread count, because ONNX Runtime checks an affinity list against
        // it: the number of entries has to match the threads it is placing.
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
    /// Hand-written because `Session` and `Environment` wrap raw C handles, which
    /// have no useful representation.
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
    /// Takes the features by value so their buffers can be lent to the engine mutably,
    /// which is what keeps inputs zero-copy: `CreateTensorWithDataAsOrtValue` wraps
    /// this memory rather than copying it.
    ///
    /// Names and shapes are collected before the values, because borrowing a name would
    /// collide with the mutable borrow each `InputValue` takes on its feature. Each
    /// shape comes from the model's own spec rather than from `[rows, width]`, because
    /// the engine checks rank: an input declared `[batch]` will not accept a
    /// two-dimensional shape even when the values are the same.
    ///
    /// Outputs come back in declared order, so they are named by zipping -- and that
    /// assumption is checked rather than trusted. If the engine ever returned them in
    /// another order, every output would be mislabelled and a caller selecting by name
    /// would silently read the wrong output.
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
///
/// These five arms are the whole engine-facing translation, which is why they belong
/// here rather than above the trait: a second engine writes its own five and shares
/// everything else.
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
/// A non-float output cannot appear here, because [`convert_specs`] refused the
/// signature when the model loaded. Handled rather than asserted, so that a model
/// whose declared and actual output types disagree gives a clear error instead of a
/// panic in a serving path.
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
/// This runs when the model loads, so an unservable model is a startup failure naming
/// the tensor and its type, not a request that fails once traffic arrives. Three things
/// are checked:
///
/// * **The element type.** An input may be `INT32`, `INT64` or `STRING` as well as a
///   float, because a feature can be passed through verbatim to feed one. An output may
///   only be `FP32` or `FP64`, because a response carries scores. That asymmetry is the
///   shape of the service, so it is enforced here where the message can say which side
///   is at fault.
/// * **The rank**, which must be `[batch]` or `[batch, width]`. Checking it once here is
///   what lets everything above this file work in rows and widths.
/// * **The leading axis**, which must be dynamic unless `mini_batch` declares a padded
///   size matching it. A model fixing it can only ever be given batches of that one
///   size, so serving one is a decision the configuration has to state.
///
/// A `[batch]` axis carries one value per row. A dynamic width is left unknown, for the
/// configuration to decide.
///
/// `mini_batch` is checked per tensor rather than across the signature, because a graph
/// can pin some axes and leave others symbolic. Every pinned axis must agree with the
/// declared size *and* be padded up to it; a dynamic one is served as it arrives.
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
                // A pinned axis is servable only if every batch is padded up to it,
                // which is exactly what `is_fixed` declares. A size without the flag
                // would send a short last batch straight into a shape mismatch, so it
                // is refused here rather than on the first odd-sized request.
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

    /// The asymmetry, at the one place it is enforced: a feature can be a category
    /// code, a score cannot.
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

    /// The gate that keeps the rest of the service to five element types. It runs at
    /// model load, so this is a startup failure rather than a request failure. INT32,
    /// INT64 and STRING are served, so what is left is the families that forced real
    /// complexity: the 16-bit floats with no protobuf field, and the integers narrower
    /// than 32 bits, which share protobuf's int32 field and so would need a range check
    /// per element.
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

    /// One function serves both call sites, so the wording has to follow. `I16` is
    /// unservable on either side; `I32` would not do here, because it is a valid
    /// *input* and only an invalid output.
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

    // ------------------------------------------------------- shape validation

    /// `[batch]` is how a scalar-per-row text or code input is exported, and the rank
    /// has to survive because an engine checks it.
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

    /// This service serves tabular models. Refusing rank 3 here is what lets everything
    /// above this file work in rows and widths.
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

    /// A model that can only take eight rows cannot serve arbitrary batches, so an
    /// undeclared pinning fails at load rather than on the first request of a different
    /// size. The message offers both ways out, because either can be the right one.
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

    /// The point of a padded mini batch: a pinned graph is servable once the deployment
    /// declares the size it was built for and says it pads up to it. The width survives the check unchanged, which
    /// is what the configuration is then resolved against.
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

    /// The one combination that would fail inside the engine on every request: the model
    /// was exported for one row count and the configuration promises another. Both
    /// numbers go in the message, because either could be the one that is wrong.
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

    /// The mistake the flag exists to catch: the right size, but batches left short. Any
    /// request whose row count is not a multiple of the size would fail inside the
    /// engine, so it is refused at load with the flag named.
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

    /// A declaration alongside a dynamic axis is legal, and means what it says: the
    /// deployment splits into that many rows per batch. Checked per tensor rather than
    /// per model because a graph can pin its inputs and leave an output symbolic. Left
    /// unpadded here, which a dynamic axis accepts and a pinned one would not.
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

    // ------------------------------------------------- against a real engine
    //
    // These need `libonnxruntime` and skip with a reason when it is absent, the
    // same rule the rest of the suite follows.

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
        match OnnxRuntimeBackend::load(&model, &ExecutionProvider::Cpu, Some(1), None, &[]) {
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

    /// Text on the *input* side is served, which is the signature a named-input
    /// configuration is built against. Outputs are float, because a response carries
    /// scores.
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

    /// `mixed_io.onnx` returns INT64 `codes` and STRING `echoed`. `onnxrt-rs` runs it
    /// happily -- its own tests do -- but a response carries FP32 or FP64 scores, so the
    /// refusal belongs here, at load, naming the output and saying it is an *output*
    /// problem, since an INT64 input is fine.
    ///
    /// A runtime that is absent, unloadable, or too old for the checked-in bindings is a
    /// reason to skip rather than to fail: none of those says anything about whether the
    /// refusal works.
    #[test]
    fn a_model_with_a_non_float_output_is_refused_when_it_loads() {
        let Some(bytes) = fixture("mixed_io.onnx") else {
            return;
        };
        let err =
            match OnnxRuntimeBackend::load(&bytes, &ExecutionProvider::Cpu, Some(1), None, &[]) {
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

    /// The design's central claim against a real engine: one call returns logits and an
    /// embedding of different widths, and the service does not need to know which kind
    /// of model it is. Cross-checked against Python onnxruntime when the fixture was
    /// generated.
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

    /// FP64 is half of what this service claims to serve, so it is worth proving
    /// against a real engine rather than only in a unit test. The fixture computes 2x +
    /// 1, exactly representable for these values.
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

    /// "number" rather than "numbers". With one input, matching positionally would send
    /// it to the right slot by luck and hide the typo entirely.
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

    /// End to end, and the case the named shape exists for: a request carrying a plain
    /// number, a categorical to one-hot, and a text feature that must reach the graph
    /// untouched.
    ///
    /// `per_feature_io.onnx` declares `tags` as `[batch, 1]`, so a feature built at the
    /// wrong rank fails here rather than being quietly reinterpreted. The fixture
    /// computes `score = age + sum(one_hot(city)) * 0.2`, and `age` has no transformation
    /// configured, so it passes through as 2.0.
    #[test]
    fn a_named_input_configuration_scores_through_a_real_model() {
        let Some(backend) = cpu_backend("per_feature_io.onnx") else {
            return;
        };

        let model_config = crate::config::ModelConfig::from_json(
            r#"{
                "model_id": "per-feature",
                "model_path": "test-data/per_feature_io.onnx",
                "vectorization_config": {
                    "feature_transformations": {
                        "city": {
                            "type": "one_hot_encoding",
                            "categories": ["berlin", "london", "paris", "rome", "tokyo"],
                            "default_val": [0, 0, 0, 0, 0]
                        }
                    },
                    "model_inputs": {
                        "age":  {"data_type": "float"},
                        "city": {"data_type": "float_array"},
                        "tags": {"data_type": "string"}
                    }
                }
            }"#,
        )
        .expect("a valid model configuration");

        let builder = crate::inference::input_builder::InputBuilder::resolve(
            model_config.vectorization_config,
            backend.inputs(),
        )
        .expect("the declared inputs match the model's signature");

        use hushar::hushar_proto::{DataType as WireValue, InputRow, data_type::DataType as Value};
        let feature = |v: Value| WireValue { data_type: Some(v) };
        let row = InputRow {
            row_id: "r1".to_owned(),
            features: [
                ("age".to_owned(), feature(Value::FloatValue(2.0))),
                (
                    "city".to_owned(),
                    feature(Value::StringValue("london".into())),
                ),
                (
                    "tags".to_owned(),
                    feature(Value::StringValue("beta".into())),
                ),
            ]
            .into_iter()
            .collect(),
        };

        let scored = crate::inference::scoring::score_features(
            &backend,
            vec![row],
            &builder,
            None,
            crate::inference::scoring::FeatureLogging::Record,
        )
        .expect("the batch should score");

        let scores = match scored.outputs[0]
            .scores
            .as_ref()
            .and_then(|s| s.score_type.as_ref())
        {
            Some(hushar::hushar_proto::score_type::ScoreType::FloatScores(f)) => f.values.clone(),
            other => panic!("expected float scores, got {other:?}"),
        };
        assert_eq!(scores.len(), 1, "the primary output is one value per row");
        assert!((scores[0] - 2.2).abs() < 1e-5, "got {scores:?}");

        assert_eq!(
            scored.logs[0].features["tags"].data_type,
            Some(Value::StringValue("beta".into()))
        );
    }

    /// The other shape, against the same engine: one concatenated FP32 input, as a
    /// tabular scorer takes. The same fixture and inputs as the direct-engine
    /// test above, so the score must match: sigmoid(sum(x) * 0.5 + 0.1).
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

        use hushar::hushar_proto::{DataType as WireValue, InputRow, data_type::DataType as Value};
        let feature = |v: f32| WireValue {
            data_type: Some(Value::FloatValue(v)),
        };
        let row = InputRow {
            row_id: "r1".to_owned(),
            features: [
                ("a".to_owned(), feature(1.0)),
                ("b".to_owned(), feature(2.0)),
                ("c".to_owned(), feature(3.0)),
            ]
            .into_iter()
            .collect(),
        };

        let scored = crate::inference::scoring::score_features(
            &backend,
            vec![row],
            &builder,
            None,
            crate::inference::scoring::FeatureLogging::Record,
        )
        .expect("the batch should score");

        let scores = match scored.outputs[0]
            .scores
            .as_ref()
            .and_then(|s| s.score_type.as_ref())
        {
            Some(hushar::hushar_proto::score_type::ScoreType::FloatScores(f)) => f.values.clone(),
            other => panic!("expected float scores, got {other:?}"),
        };
        assert!((scores[0] - 0.9568927).abs() < 1e-5, "got {scores:?}");
    }

    /// End to end through a real pinned graph. `sigmoid_model_3_batch2.onnx` fixes its
    /// leading dimension at 2, so ONNX Runtime refuses any other row count outright --
    /// which makes this the test that the padding actually satisfies the engine, and not
    /// merely the service's own bookkeeping.
    ///
    /// The fixture also makes the trimming *observable* rather than just asserted. Its
    /// score is `sigmoid(0.5 * sum(row) + 0.1)`, so a row of ones scores ~0.832 while a
    /// padding row, which is all zeros, scores ~0.525. A padded row leaking into a
    /// response would show up as a value no real row here can produce.
    #[test]
    fn a_pinned_graph_serves_a_request_of_any_row_count() {
        let Some(model) = fixture("sigmoid_model_3_batch2.onnx") else {
            return;
        };
        let backend = match OnnxRuntimeBackend::load(
            &model,
            &ExecutionProvider::Cpu,
            Some(1),
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
            crate::inference::input_builder::InputBuilder::resolve(Some(config), backend.inputs())
                .expect("three features fill the model's one 3-wide input");

        use hushar::hushar_proto::{DataType as WireValue, InputRow, data_type::DataType as Value};
        let ones = |id: &str| InputRow {
            row_id: id.to_owned(),
            features: ["a", "b", "c"]
                .into_iter()
                .map(|n| {
                    (
                        n.to_owned(),
                        WireValue {
                            data_type: Some(Value::FloatValue(1.0)),
                        },
                    )
                })
                .collect(),
        };
        let float_scores = |row: &hushar::hushar_proto::OutputRow| match row
            .scores
            .as_ref()
            .and_then(|s| s.score_type.as_ref())
        {
            Some(hushar::hushar_proto::score_type::ScoreType::FloatScores(f)) => f.values.clone(),
            other => panic!("expected float scores, got {other:?}"),
        };

        const REAL: f32 = 0.832_018_4; // sigmoid(0.5 * 3 + 0.1)
        const PAD: f32 = 0.524_979_2; // sigmoid(0.5 * 0 + 0.1), a padding row

        // One row, three rows, four rows: under, over-and-ragged, and an exact multiple.
        // Every one of them is a row count the graph itself cannot accept.
        for count in [1usize, 3, 4] {
            let rows: Vec<InputRow> = (0..count).map(|i| ones(&format!("r{i}"))).collect();
            let scored = crate::inference::scoring::score_features(
                &backend,
                rows,
                &builder,
                Some(crate::inference::scoring::MiniBatch {
                    size: 2,
                    is_fixed: true,
                }),
                crate::inference::scoring::FeatureLogging::Record,
            )
            .unwrap_or_else(|e| panic!("{count} rows against a pinned graph: {e}"));

            assert_eq!(
                scored.outputs.len(),
                count,
                "{count} rows in must be {count} rows out"
            );
            assert_eq!(scored.logs.len(), count, "the log must not carry padding");

            for (i, out) in scored.outputs.iter().enumerate() {
                assert_eq!(out.row_id, format!("r{i}"), "row order must survive");
                let s = float_scores(out);
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
