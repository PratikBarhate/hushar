// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Turning a request into model outputs.
//!
//! One way in. [`score_features`] takes named feature values per row and hands them
//! to an [`InputBuilder`], which was settled against the model's signature at
//! startup. Everything after that is shared: one engine call, one output-checking
//! rule, one set of timings.
//!
//! There is no second entry point taking tensors. A caller that has already
//! vectorized sends its vector as one feature whose value is a `FloatArray`, which a
//! pass-through transformation hands to the model unchanged. That keeps one request
//! shape and removes a class of malformed request outright: no shape disagreeing
//! with the element count, and no input name matching nothing.
//!
//! The engine is reached only through [`InferenceBackend`], so none of this changes
//! for ONNX Runtime, Neuron, or anything added later.

use hushar::hushar_proto::{
    DoubleArray, FloatArray, InferenceLogRow, InputRow, OutputRow, ScoreType,
    score_type::ScoreType as Scores,
};
use std::collections::HashMap;
use std::time::Instant;

use crate::inference::InferenceMicros;
use crate::inference::backend::InferenceBackend;
use crate::inference::batch::{InputBatch, Output, ScoreData};
use crate::inference::input_builder::InputBuilder;

/// One scored batch.
///
/// Fields:
/// - `outputs` — one row per input row, in the same order.
/// - `logs` — the same rows for the inference log, carrying the features as received.
/// - `timings` — how long each stage took.
///
/// A struct rather than a tuple because it carries three things, and
/// `(Vec<OutputRow>, Vec<InferenceLogRow>, InferenceMicros)` at a call site says nothing
/// about which is which.
#[derive(Debug)]
pub struct ScoredBatch {
    pub outputs: Vec<OutputRow>,
    pub logs: Vec<InferenceLogRow>,
    pub timings: InferenceMicros,
}

impl ScoredBatch {
    /// The result of scoring nothing.
    fn empty() -> Self {
        Self {
            outputs: Vec::new(),
            logs: Vec::new(),
            timings: InferenceMicros {
                vec_time: 0,
                tensor_time: 0,
                inference_time: 0,
            },
        }
    }
}

/// The `row_id` given to a padding row.
///
/// It never reaches a caller or a log -- padding rows are dropped before either -- so
/// this exists to make a padded row obvious in a debugger or a backend that prints its
/// input, rather than to be matched on.
const PAD_ROW_ID: &str = "__hushar_pad__";

/// Whether a request's features are kept for the inference log.
///
/// Decided per request *before* scoring, by
/// [`Sampler`](crate::io::Sampler), which is what makes sampling a saving rather
/// than a filter: a request that will not be logged never has its feature values moved
/// anywhere, and `Skip` allocates no log rows at all.
///
/// A `bool` would read as `score_features(.., true)` at the call site, which says
/// nothing about which `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureLogging {
    /// Keep the features and scores for this request.
    Record,
    /// Score it and keep nothing.
    Skip,
}

/// Scores rows of named features.
///
/// Each row's features are recorded for the inference log **as received, before any
/// transformation**, and the whole map is kept -- including keys the configuration
/// ignores -- so the log shows what the caller actually sent rather than what was
/// made of it. Debugging a bad prediction means knowing what came in.
///
/// # Nothing is copied to log it
///
/// `rows` is taken by value, and the features are *moved* out of it into the log rows
/// once the inputs have been built. Feature values are the largest thing a request
/// carries -- a map per row, holding strings and arrays -- and this function used to
/// deep-clone all of it before building anything, so that the originals were still
/// around afterwards. Owning the rows means the log gets the values themselves. The
/// only thing copied per row is its `row_id`, which a response and a log row both
/// need.
///
/// Every check that does not need a request has already run, when the builder was
/// resolved against the model's signature at startup. What is left here is what only
/// a request can be wrong about: a feature of the wrong type for a verbatim input,
/// and a row that produces a different number of values from its neighbours.
///
/// # Fixed batch size
///
/// `fixed_batch_size` is the row count the model's graph is pinned to, when it is
/// pinned. A request of any size is still served: rows are cut into batches of that
/// many, and the final short batch is padded to it. So a pinned model is a property of
/// the deployment rather than a constraint on callers.
///
/// ```text
/// fixed_batch_size: 4,  request of 6 rows
///
///   r0 r1 r2 r3 │ r4 r5 ·· ··      ·· = padding row, all defaults
///   └── run 1 ──┘ └── run 2 ──┘
///        ▼             ▼
///   s0 s1 s2 s3    s4 s5 xx xx      xx = scored, then discarded
///   └──────── 6 rows out ───┘
/// ```
///
/// A padding row carries no features, so every input takes the same path as a request
/// that omitted that feature: its transformation's `default_val`, or zeros and empty
/// strings for the inputs passed through verbatim. Nothing new has to be correct for
/// padding to be correct.
///
/// **Two things this costs, both worth stating.** Padding is real work -- one row sent
/// to a model pinned at 32 costs a batch of 32 -- so a pinned deployment should pin to
/// a size near its traffic, not an arbitrary one. And the batches run in sequence on
/// this thread, which keeps one inference in flight per request as
/// [`spawn_blocking`](tokio::task::spawn_blocking) intends; a request of 100 rows
/// against a size of 1 is 100 engine calls, and it will feel like it.
///
/// `None` runs the request as a single batch of whatever size it is, which is what a
/// model with a dynamic row axis wants.
pub fn score_features(
    backend: &dyn InferenceBackend,
    rows: Vec<InputRow>,
    builder: &InputBuilder,
    fixed_batch_size: Option<usize>,
    logging: FeatureLogging,
) -> Result<ScoredBatch, crate::inference::InferenceError> {
    if rows.is_empty() {
        return Ok(ScoredBatch::empty());
    }

    let total = rows.len();
    // A dynamic deployment is the degenerate case of one batch holding everything, so
    // there is one code path below rather than a branch that could diverge.
    let size = fixed_batch_size.unwrap_or(total).max(1);

    let mut scored = ScoredBatch {
        outputs: Vec::with_capacity(total),
        logs: match logging {
            FeatureLogging::Record => Vec::with_capacity(total),
            FeatureLogging::Skip => Vec::new(),
        },
        timings: InferenceMicros {
            vec_time: 0,
            tensor_time: 0,
            inference_time: 0,
        },
    };

    // Consumed rather than indexed, so each chunk owns its rows and the features can be
    // handed to the log instead of copied for it.
    let mut remaining = rows.into_iter();
    loop {
        let chunk: Vec<InputRow> = remaining.by_ref().take(size).collect();
        if chunk.is_empty() {
            break;
        }
        score_chunk(backend, chunk, builder, size, &mut scored, logging)?;
    }

    Ok(scored)
}

/// Scores one batch, padding it to `size` first if it is short, and appends the real
/// rows' results to `scored`.
///
/// Padding is appended to the chunk this function already owns, so a short batch costs
/// only the padding rows -- where an earlier version cloned every real row to make room
/// for them. The padding is then truncated away before the rows are handed on, which is
/// what keeps a synthetic row out of both the response and the log without anything
/// downstream having to know padding exists.
///
/// The timings accumulate across batches, so a request's `InferenceTime` is the engine
/// time it actually cost. That is the number worth having: two batches of 4 really do
/// spend twice as long in the engine as one, and a metric that reported only the last
/// batch would hide exactly the cost this function introduces.
fn score_chunk(
    backend: &dyn InferenceBackend,
    mut chunk: Vec<InputRow>,
    builder: &InputBuilder,
    size: usize,
    scored: &mut ScoredBatch,
    logging: FeatureLogging,
) -> Result<(), crate::inference::InferenceError> {
    let vec_start = Instant::now();

    let real = chunk.len();
    if real < size {
        chunk.resize(
            size,
            InputRow {
                row_id: PAD_ROW_ID.to_owned(),
                features: HashMap::new(),
            },
        );
    }
    let sent = chunk.len();

    let inputs = builder.build(&chunk)?;
    let build_start = Instant::now();

    // The engine has what it needs, and `InputBatch` owns its buffers, so the padding
    // rows are of no further use to anyone.
    chunk.truncate(real);

    run_batch(
        backend,
        chunk,
        inputs,
        sent,
        vec_start,
        build_start,
        scored,
        logging,
    )
}

/// Runs the model and turns its primary output into rows.
///
/// The log stores scores as floats whatever the model's precision, because it is read
/// by batch jobs and training pipelines that want one stable column.
///
/// `sent` is how many rows the engine was given, which is `kept.len()` plus any
/// padding. The engine is held to returning one row per row it was *sent*, and only the
/// first `kept.len()` are read back -- so a padded row is scored and dropped, never
/// reaching a response or a log. Checking against `sent` rather than against the rows
/// kept is what makes that distinction, and it is why a model returning the wrong row
/// count is still caught.
///
/// `kept` is consumed: each row's `features` map moves into its log row, and its
/// `row_id` moves into whichever of the two does not need a copy.
///
/// # Only the first output is returned
///
/// A model may declare several; the first is the one scored. That covers every model
/// family this serves -- class probabilities, a regression value, a flattened
/// forecast horizon, an embedding -- each of which is one numeric vector per row. A
/// model whose extra outputs matter would need a richer response, and none does.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    backend: &dyn InferenceBackend,
    kept: Vec<InputRow>,
    inputs: InputBatch,
    sent: usize,
    stage_start: Instant,
    build_start: Instant,
    scored: &mut ScoredBatch,
    logging: FeatureLogging,
) -> Result<(), crate::inference::InferenceError> {
    let inference_start = Instant::now();
    let produced = backend.run(inputs)?;
    let inference_end = Instant::now();

    let primary = produced.first().ok_or("model returned no outputs at all")?;
    if produced.rows() != sent {
        return Err(format!(
            "model output {:?} has {} rows for a batch of {sent} rows",
            primary.name,
            produced.rows(),
        )
        .into());
    }

    for (i, row) in kept.into_iter().enumerate() {
        let scores = primary.row_as_f32(i).ok_or_else(|| {
            format!(
                "model output {:?} returned no values for row {i}",
                primary.name
            )
        })?;
        let score_type = Some(to_score_type(primary, i, &scores));

        match logging {
            // The id is the one thing both rows need, so it is the one thing copied.
            FeatureLogging::Record => {
                scored.outputs.push(OutputRow {
                    row_id: row.row_id.clone(),
                    scores: score_type,
                });
                scored.logs.push(InferenceLogRow {
                    row_id: row.row_id,
                    features: row.features,
                    scores,
                });
            }
            FeatureLogging::Skip => scored.outputs.push(OutputRow {
                row_id: row.row_id,
                scores: score_type,
            }),
        }
    }

    scored.timings.vec_time += (build_start - stage_start).as_micros();
    scored.timings.tensor_time += (inference_start - build_start).as_micros();
    scored.timings.inference_time += (inference_end - inference_start).as_micros();

    Ok(())
}

/// One row's scores in the model's own precision.
///
/// `FP64` is kept as doubles rather than narrowed, which is the whole reason
/// [`ScoreType`] exists: a model asked for double precision should not have it thrown
/// away on the way out. `f32` values are already to hand, so they are reused.
///
/// Infallible, and that is a property of the type rather than of this code: an
/// [`Output`] is `FP32` or `FP64` by construction, so there is no third case to report.
/// The caller has already established that this row exists.
fn to_score_type(primary: &Output, row: usize, as_f32: &[f32]) -> ScoreType {
    let score_type = match &primary.data {
        ScoreData::F64(_) => Scores::DoubleScores(DoubleArray {
            values: primary.row_f64(row).unwrap_or_default().to_vec(),
        }),
        ScoreData::F32(_) => Scores::FloatScores(FloatArray {
            values: as_f32.to_vec(),
        }),
    };
    ScoreType {
        score_type: Some(score_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::vectorization_config::VectorizationConfig;
    use crate::inference::backend::test_support::{
        DoubleBackend, FailingBackend, MultiOutputBackend, PerFeatureBackend, RecordingBackend,
        SummingBackend,
    };
    use hushar::hushar_proto::DataType as WireValue;
    use hushar::hushar_proto::data_type::DataType as Value;

    /// A builder for `backend`, from the given configuration JSON.
    fn builder_for(backend: &dyn InferenceBackend, json: &str) -> InputBuilder {
        let config = VectorizationConfig::from_json(json).expect("valid configuration");
        InputBuilder::resolve(Some(config), backend.inputs()).expect("fits the model")
    }

    /// A vectorised configuration of `width` pass-through features named `f0..`.
    fn vector_config(width: usize, data_type: &str) -> String {
        let names: Vec<String> = (0..width).map(|i| format!("\"f{i}\"")).collect();
        format!(
            r#"{{"data_type": "{data_type}", "feature_order": [{}]}}"#,
            names.join(", ")
        )
    }

    fn row(row_id: &str, features: &[(&str, Value)]) -> InputRow {
        InputRow {
            row_id: row_id.to_owned(),
            features: features
                .iter()
                .map(|(name, value)| {
                    (
                        (*name).to_owned(),
                        WireValue {
                            data_type: Some(value.clone()),
                        },
                    )
                })
                .collect(),
        }
    }

    /// `rows` rows whose feature `f0..fn` all hold `value`.
    fn numeric_rows(rows: usize, width: usize, value: f32) -> Vec<InputRow> {
        (0..rows)
            .map(|r| {
                let features: Vec<(String, Value)> = (0..width)
                    .map(|i| (format!("f{i}"), Value::FloatValue(value)))
                    .collect();
                InputRow {
                    row_id: format!("row-{r}"),
                    features: features
                        .into_iter()
                        .map(|(name, value)| {
                            (
                                name,
                                WireValue {
                                    data_type: Some(value),
                                },
                            )
                        })
                        .collect(),
                }
            })
            .collect()
    }

    fn float_scores(output: &OutputRow) -> Vec<f32> {
        match output.scores.as_ref().and_then(|s| s.score_type.as_ref()) {
            Some(Scores::FloatScores(f)) => f.values.clone(),
            other => panic!("expected float scores, got {other:?}"),
        }
    }

    fn double_scores(output: &OutputRow) -> Vec<f64> {
        match output.scores.as_ref().and_then(|s| s.score_type.as_ref()) {
            Some(Scores::DoubleScores(d)) => d.values.clone(),
            other => panic!("expected double scores, got {other:?}"),
        }
    }

    // ------------------------------------------------------------ the happy path

    #[test]
    fn the_feature_path_scores_every_row_and_reports_timings() {
        let backend = SummingBackend::new(3, 2);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            numeric_rows(4, 3, 1.0),
            &builder,
            None,
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(scored.outputs.len(), 4);
        assert_eq!(scored.logs.len(), 4);
        assert!(
            scored.timings.inference_time > 0 || scored.timings.vec_time > 0,
            "some stage must have taken measurable time"
        );
    }

    /// The backend sums each row, so a transposition between rows changes the answer --
    /// which a constant-returning backend could not catch.
    #[test]
    fn each_row_receives_its_own_scores() {
        let backend = SummingBackend::new(2, 1);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let rows = vec![
            row(
                "a",
                &[
                    ("f0", Value::FloatValue(1.0)),
                    ("f1", Value::FloatValue(1.0)),
                ],
            ),
            row(
                "b",
                &[
                    ("f0", Value::FloatValue(5.0)),
                    ("f1", Value::FloatValue(5.0)),
                ],
            ),
        ];
        let scored =
            score_features(&backend, rows, &builder, None, FeatureLogging::Record).expect("scored");

        assert_eq!(scored.outputs[0].row_id, "a");
        assert_eq!(float_scores(&scored.outputs[0]), vec![2.0]);
        assert_eq!(scored.outputs[1].row_id, "b");
        assert_eq!(float_scores(&scored.outputs[1]), vec![10.0]);
    }

    /// Including a key the configuration ignores: the log is for debugging what
    /// arrived, not for reporting what vectorization made of it.
    #[test]
    fn the_log_records_the_features_as_received() {
        let backend = SummingBackend::new(1, 1);
        let builder = builder_for(&backend, &vector_config(1, "float"));

        let rows = vec![row(
            "a",
            &[
                ("f0", Value::FloatValue(2.0)),
                ("ignored", Value::StringValue("kept anyway".into())),
            ],
        )];
        let scored =
            score_features(&backend, rows, &builder, None, FeatureLogging::Record).expect("scored");

        let logged = &scored.logs[0].features;
        assert_eq!(logged.len(), 2, "both keys survive into the log");
        assert!(logged.contains_key("ignored"));
        assert_eq!(scored.logs[0].scores, vec![2.0]);
    }

    /// `FailingBackend` fails whenever it is called, so reaching it would fail.
    #[test]
    fn an_empty_batch_is_not_sent_to_the_engine() {
        let backend = FailingBackend;
        let builder = builder_for(&backend, &vector_config(1, "float"));

        let scored = score_features(&backend, Vec::new(), &builder, None, FeatureLogging::Record)
            .expect("nothing to do");
        assert!(scored.outputs.is_empty());
        assert!(scored.logs.is_empty());
        assert_eq!(scored.timings.inference_time, 0);
    }

    // ------------------------------------------------------------- sampling

    /// What an unsampled request costs: a response, and nothing else. The features are
    /// dropped with the request rather than copied into a log nobody asked for.
    #[test]
    fn a_skipped_request_is_scored_and_logs_nothing() {
        let backend = SummingBackend::new(2, 1);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let scored = score_features(
            &backend,
            numeric_rows(3, 2, 2.0),
            &builder,
            None,
            FeatureLogging::Skip,
        )
        .expect("scored");

        assert_eq!(scored.outputs.len(), 3, "every row is still answered");
        assert_eq!(float_scores(&scored.outputs[0]), vec![4.0]);
        assert!(
            scored.logs.is_empty(),
            "a skipped request should carry no log rows at all"
        );
    }

    /// Sampling must not be observable by a caller. Anything else would make the lever
    /// a correctness decision rather than an observability one.
    #[test]
    fn skipping_the_log_does_not_change_the_response() {
        let recorded = {
            let backend = SummingBackend::new(2, 1);
            let builder = builder_for(&backend, &vector_config(2, "float"));
            score_features(
                &backend,
                numeric_rows(4, 2, 1.5),
                &builder,
                Some(3),
                FeatureLogging::Record,
            )
            .expect("scored")
        };
        let skipped = {
            let backend = SummingBackend::new(2, 1);
            let builder = builder_for(&backend, &vector_config(2, "float"));
            score_features(
                &backend,
                numeric_rows(4, 2, 1.5),
                &builder,
                Some(3),
                FeatureLogging::Skip,
            )
            .expect("scored")
        };

        assert_eq!(recorded.outputs.len(), skipped.outputs.len());
        for (a, b) in recorded.outputs.iter().zip(skipped.outputs.iter()) {
            assert_eq!(a.row_id, b.row_id);
            assert_eq!(float_scores(a), float_scores(b));
        }
        assert_eq!(recorded.logs.len(), 4);
        assert!(skipped.logs.is_empty());
    }

    /// The values in the log are the request's own, moved rather than copied. A test
    /// cannot observe an address, but it can observe that nothing was lost on the way:
    /// a string feature arrives intact, which a move preserves and a partial copy would
    /// not.
    #[test]
    fn the_log_receives_the_requests_own_feature_values() {
        let backend = PerFeatureBackend::new();
        let builder = builder_for(
            &backend,
            r#"{
                "model_inputs": {
                    "age": {"data_type": "float"},
                    "tags": {"data_type": "string"}
                }
            }"#,
        );

        let long_text = "x".repeat(4096);
        let rows = vec![row(
            "a",
            &[
                ("age", Value::FloatValue(1.0)),
                ("tags", Value::StringValue(long_text.clone())),
            ],
        )];
        let scored =
            score_features(&backend, rows, &builder, None, FeatureLogging::Record).expect("scored");

        assert_eq!(
            scored.logs[0].features["tags"].data_type,
            Some(Value::StringValue(long_text)),
            "the whole value should arrive in the log"
        );
    }

    #[test]
    fn an_engine_failure_is_propagated() {
        let backend = FailingBackend;
        let builder = builder_for(&backend, &vector_config(1, "float"));

        let err = score_features(
            &backend,
            numeric_rows(1, 1, 1.0),
            &builder,
            None,
            FeatureLogging::Record,
        )
        .expect_err("the engine failed");
        assert!(err.to_string().contains("engine exploded"), "got {err}");
    }

    // ------------------------------------------------------- named-input models

    /// Two inputs of different element types in one call: the case a single float
    /// matrix cannot express. score = age + total length of the row's tags.
    #[test]
    fn a_named_input_model_is_fed_from_a_feature_map() {
        let backend = PerFeatureBackend::new();
        let builder = builder_for(
            &backend,
            r#"{
                "model_inputs": {
                    "age": {"data_type": "float"},
                    "tags": {"data_type": "string"}
                }
            }"#,
        );

        let rows = vec![
            row(
                "a",
                &[
                    ("age", Value::FloatValue(10.0)),
                    ("tags", Value::StringValue("abc".into())),
                ],
            ),
            row(
                "b",
                &[
                    ("age", Value::FloatValue(20.0)),
                    ("tags", Value::StringValue("de".into())),
                ],
            ),
        ];
        let scored =
            score_features(&backend, rows, &builder, None, FeatureLogging::Record).expect("scored");

        assert_eq!(float_scores(&scored.outputs[0]), vec![13.0]);
        assert_eq!(float_scores(&scored.outputs[1]), vec![22.0]);
    }

    #[test]
    fn a_text_feature_is_logged_as_the_text_it_was() {
        let backend = PerFeatureBackend::new();
        let builder = builder_for(
            &backend,
            r#"{
                "model_inputs": {
                    "age": {"data_type": "float"},
                    "tags": {"data_type": "string"}
                }
            }"#,
        );

        let rows = vec![row(
            "a",
            &[
                ("age", Value::FloatValue(1.0)),
                ("tags", Value::StringValue("london".into())),
            ],
        )];
        let scored =
            score_features(&backend, rows, &builder, None, FeatureLogging::Record).expect("scored");

        assert_eq!(
            scored.logs[0].features["tags"].data_type,
            Some(Value::StringValue("london".into())),
            "the log keeps text as text rather than as whatever number it became"
        );
    }

    /// A partial request is normal on a serving path. The empty string reaches the
    /// model, and the row still gets a score.
    #[test]
    fn an_absent_text_feature_scores_rather_than_failing() {
        let backend = PerFeatureBackend::new();
        let builder = builder_for(
            &backend,
            r#"{
                "model_inputs": {
                    "age": {"data_type": "float"},
                    "tags": {"data_type": "string"}
                }
            }"#,
        );

        let rows = vec![row("a", &[("age", Value::FloatValue(3.0))])];
        let scored = score_features(&backend, rows, &builder, None, FeatureLogging::Record)
            .expect("a partial row scores");
        assert_eq!(
            float_scores(&scored.outputs[0]),
            vec![3.0],
            "an empty tag adds no length"
        );
    }

    #[test]
    fn a_feature_of_the_wrong_type_for_a_verbatim_input_is_refused() {
        let backend = PerFeatureBackend::new();
        let builder = builder_for(
            &backend,
            r#"{
                "model_inputs": {
                    "age": {"data_type": "float"},
                    "tags": {"data_type": "string"}
                }
            }"#,
        );

        let rows = vec![row(
            "r9",
            &[
                ("age", Value::FloatValue(1.0)),
                ("tags", Value::LongValue(7)),
            ],
        )];
        let err = score_features(&backend, rows, &builder, None, FeatureLogging::Record)
            .expect_err("a long is not text");
        let m = err.to_string();
        assert!(m.contains("r9") && m.contains("tags"), "got {m}");
    }

    // ------------------------------------------------------------- precision

    #[test]
    fn a_single_precision_model_returns_float_scores() {
        let backend = SummingBackend::new(2, 1);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let scored = score_features(
            &backend,
            numeric_rows(1, 2, 1.5),
            &builder,
            None,
            FeatureLogging::Record,
        )
        .expect("scored");
        assert!(matches!(
            scored.outputs[0].scores.as_ref().unwrap().score_type,
            Some(Scores::FloatScores(_))
        ));
    }

    /// 0.1 has no exact f32, so this is the test that nothing narrowed on either the
    /// way in or the way out.
    #[test]
    fn a_double_precision_model_returns_double_scores_unnarrowed() {
        let backend = DoubleBackend::new(1);
        let builder = builder_for(&backend, &vector_config(1, "double"));

        let rows = vec![row("a", &[("f0", Value::DoubleValue(0.1))])];
        let scored =
            score_features(&backend, rows, &builder, None, FeatureLogging::Record).expect("scored");
        assert_eq!(double_scores(&scored.outputs[0]), vec![0.1]);
    }

    #[test]
    fn double_scores_are_read_from_the_right_row() {
        let backend = DoubleBackend::new(1);
        let builder = builder_for(&backend, &vector_config(1, "double"));

        let rows = vec![
            row("a", &[("f0", Value::DoubleValue(0.25))]),
            row("b", &[("f0", Value::DoubleValue(0.75))]),
        ];
        let scored =
            score_features(&backend, rows, &builder, None, FeatureLogging::Record).expect("scored");
        assert_eq!(double_scores(&scored.outputs[0]), vec![0.25]);
        assert_eq!(double_scores(&scored.outputs[1]), vec![0.75]);
    }

    // ------------------------------------------------------------ many outputs

    /// `scores` is [sum, -sum]; the embedding and the total are not returned.
    #[test]
    fn only_the_models_first_output_is_returned() {
        let backend = MultiOutputBackend::new(2);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let scored = score_features(
            &backend,
            numeric_rows(2, 2, 1.5),
            &builder,
            None,
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(float_scores(&scored.outputs[0]), vec![3.0, -3.0]);
        assert_eq!(scored.logs[0].scores, vec![3.0, -3.0]);
    }

    // ---- Fixed batch size -------------------------------------------------------
    //
    // A pinned model can only be given one batch size, so the service makes any request
    // fit: rows are cut into batches of that size and the last one is padded. What these
    // tests hold onto is that the adjustment is invisible from outside -- the caller gets
    // one row out per row in, with the same scores it would have got unpinned.
    //
    // `RecordingBackend` reports the size of every batch it ran, which is the only way to
    // tell "padded to four" from "ran three rows" -- both return three rows.

    /// `rows` rows of `width` features, where row `r` holds `r` in every feature.
    ///
    /// Its sum is therefore `width * r`, so a row's scores name the row they came from.
    /// A padding row holds no features and sums to 0, so it is distinguishable from
    /// every real row but the first.
    fn ramp_rows(rows: usize, width: usize) -> Vec<InputRow> {
        (0..rows)
            .map(|r| InputRow {
                row_id: format!("row-{r}"),
                features: (0..width)
                    .map(|i| {
                        (
                            format!("f{i}"),
                            WireValue {
                                data_type: Some(Value::FloatValue(r as f32)),
                            },
                        )
                    })
                    .collect(),
            })
            .collect()
    }

    /// The reason this path exists: a pinned model serves a request of any size.
    #[test]
    fn a_request_larger_than_the_fixed_size_is_split_into_batches_of_it() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            ramp_rows(6, 3),
            &builder,
            Some(4),
            FeatureLogging::Record,
        )
        .expect("six rows against a size of four");

        // Two engine calls of exactly four rows; the second held two real rows and two
        // padding rows.
        assert_eq!(backend.batches(), vec![4, 4]);
        // Six rows in, six rows out, in order, each with its own scores.
        assert_eq!(scored.outputs.len(), 6);
        for r in 0..6 {
            assert_eq!(scored.outputs[r].row_id, format!("row-{r}"));
            assert_eq!(float_scores(&scored.outputs[r]), vec![3.0 * r as f32]);
        }
    }

    #[test]
    fn a_request_smaller_than_the_fixed_size_is_padded_up_to_it() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            ramp_rows(1, 3),
            &builder,
            Some(4),
            FeatureLogging::Record,
        )
        .expect("one row against a size of four");

        assert_eq!(backend.batches(), vec![4]);
        assert_eq!(scored.outputs.len(), 1);
        assert_eq!(scored.outputs[0].row_id, "row-0");
    }

    #[test]
    fn an_exact_multiple_of_the_fixed_size_is_split_evenly() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            ramp_rows(8, 3),
            &builder,
            Some(4),
            FeatureLogging::Record,
        )
        .expect("eight rows against a size of four");

        assert_eq!(backend.batches(), vec![4, 4]);
        assert_eq!(scored.outputs.len(), 8);
        assert_eq!(float_scores(&scored.outputs[7]), vec![21.0]);
    }

    /// The padded rows are scored -- the engine has no way not to -- and then dropped.
    /// A caller must never see one, and neither must the inference log, which feeds
    /// training pipelines that would treat a synthetic row as real data.
    #[test]
    fn padding_reaches_neither_the_response_nor_the_log() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            ramp_rows(5, 3),
            &builder,
            Some(4),
            FeatureLogging::Record,
        )
        .expect("five rows against a size of four");

        assert_eq!(backend.batches(), vec![4, 4]);
        assert_eq!(scored.outputs.len(), 5);
        assert_eq!(scored.logs.len(), 5);
        assert!(!scored.outputs.iter().any(|o| o.row_id == PAD_ROW_ID));
        assert!(!scored.logs.iter().any(|l| l.row_id == PAD_ROW_ID));
    }

    /// The size the benchmark's pinned models use, and the worst case for padding: one
    /// engine call per row.
    #[test]
    fn a_fixed_size_of_one_runs_a_batch_per_row() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            ramp_rows(3, 3),
            &builder,
            Some(1),
            FeatureLogging::Record,
        )
        .expect("three rows against a size of one");

        assert_eq!(backend.batches(), vec![1, 1, 1]);
        assert_eq!(scored.outputs.len(), 3);
        assert_eq!(float_scores(&scored.outputs[2]), vec![6.0]);
    }

    /// No declaration means no chunking, which is what a dynamic model wants and what
    /// this function did before it knew about batch sizes at all.
    #[test]
    fn a_dynamic_deployment_runs_the_whole_request_as_one_batch() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            ramp_rows(6, 3),
            &builder,
            None,
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(backend.batches(), vec![6]);
        assert_eq!(scored.outputs.len(), 6);
    }

    /// An empty request runs no model, pinned or not -- so a pinned deployment does not
    /// turn "nothing to score" into a batch of pure padding.
    #[test]
    fn an_empty_request_runs_no_batch_even_when_pinned() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            Vec::new(),
            &builder,
            Some(4),
            FeatureLogging::Record,
        )
        .expect("nothing");

        assert!(backend.batches().is_empty());
        assert!(scored.outputs.is_empty());
        assert!(scored.logs.is_empty());
    }

    /// The property that makes padding safe to do at all: a row's scores do not depend
    /// on what shared its batch. Scoring the same rows whole and in padded batches of
    /// two must agree row for row -- if padding leaked into a neighbour, or a batch were
    /// stitched back in the wrong order, this is what would catch it.
    #[test]
    fn chunking_and_padding_do_not_change_a_rows_scores() {
        let dynamic = RecordingBackend::new(3, 1);
        let dynamic_builder = builder_for(&dynamic, &vector_config(3, "float"));
        let whole = score_features(
            &dynamic,
            ramp_rows(7, 3),
            &dynamic_builder,
            None,
            FeatureLogging::Record,
        )
        .expect("scored");

        let pinned = RecordingBackend::new(3, 1);
        let pinned_builder = builder_for(&pinned, &vector_config(3, "float"));
        let split = score_features(
            &pinned,
            ramp_rows(7, 3),
            &pinned_builder,
            Some(2),
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(dynamic.batches(), vec![7]);
        // Three full batches and a fourth holding one real row and one padding row.
        assert_eq!(pinned.batches(), vec![2, 2, 2, 2]);

        assert_eq!(whole.outputs.len(), split.outputs.len());
        for (w, s) in whole.outputs.iter().zip(split.outputs.iter()) {
            assert_eq!(w.row_id, s.row_id);
            assert_eq!(float_scores(w), float_scores(s));
        }
    }
}
