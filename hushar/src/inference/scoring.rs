// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Turning a request into model outputs.
//!
//! [`score_features`] takes named feature columns and hands them to an [`InputBuilder`]
//! resolved against the model's signature at startup. Every feature is transformed and
//! concatenated into the model's one input, so a caller that has already vectorized
//! sends its vector as one column and a pass-through transformation hands it to the
//! model unchanged.
//!
//! The engine is reached only through [`InferenceBackend`].

use hushar::hushar_proto::{DoubleArray, FloatArray, ScoreType, score_type::ScoreType as Scores};
use std::ops::Range;
use std::sync::OnceLock;
use std::time::Instant;

use crate::inference::RequestReport;
use crate::inference::backend::InferenceBackend;
use crate::inference::batch::{InputBatch, Output, ScoreData};
use crate::inference::execution_plan::ExecutionPlan;
use crate::inference::input_builder::{InputBuilder, RequestColumns};

/// One scored batch.
///
/// # Fields
///
/// * `scores` - every row's scores back to back, row-major, in the model's own precision
/// * `timings` - how long each stage took
///
/// The features are not here: the builder only borrows them, so they stay in the
/// request's columns and move into the log from there.
#[derive(Debug)]
pub struct ScoredBatch {
    pub scores: ScoreData,
    pub timings: RequestReport,
}

impl ScoredBatch {
    /// The result of scoring nothing.
    pub(crate) fn empty() -> Self {
        Self {
            scores: ScoreData::F32(Vec::new()),
            timings: RequestReport::default(),
        }
    }

    /// Records how full this request's calls were.
    pub(crate) fn set_fill(&mut self, kept: usize, sent: usize) {
        self.timings.set_fill(kept, sent);
    }
}

/// Whether a request's features are kept for the inference log.
///
/// Decided per request before scoring, by [`Sampler`](crate::io::Sampler): a request that
/// will not be logged never has its feature values moved anywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureLogging {
    /// Keep the features and scores for this request.
    Record,
    /// Score it and keep nothing.
    Skip,
}

/// How a request's rows are cut into batches for the engine.
///
/// # Fields
///
/// * `size` - rows per batch
/// * `is_fixed` - pad the last batch up to `size` rather than sending it short
///
/// Constructed from the model configuration's `mini_batch_size` and `is_fixed`, which is
/// where both are documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniBatch {
    pub size: usize,
    pub is_fixed: bool,
}

/// The pool the mini batches run on.
///
/// A request's rows are cut into batches and scored at the same time, one thread each
/// from this pool, which the calling thread joins for the duration. Without a
/// `mini_batch` a request is one batch and never reaches the pool.
///
/// ```text
/// mini_batch { size: 4, is_fixed: false },  request of 6 rows
///
///   r0 r1 r2 r3 │ r4 r5
///   └── run 1 ──┘ └ run 2 ┘     both in flight together
///        ▼            ▼
///   s0 s1 s2 s3    s4 s5
///   └────── 6 rows out ─┘       order preserved, whichever finishes first
/// ```
///
/// Why splitting lowers latency is in [`crate::config::model_config::ModelConfig`].
static FAN_OUT: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Installs the mini-batch pool. Call once, before serving.
///
/// Separate from rayon's global pool because the width is chosen rather than inherited:
/// `main` sizes it to what the loaded engines can actually overlap and counts it against
/// the thread budget, where rayon's global default would spawn one thread per core and be
/// invisible to that arithmetic. No thread here is bound to a core -- the service sets no
/// CPU affinity anywhere, and placement is left to the scheduler.
///
/// # Errors
///
/// Returns the pool back if one is already installed.
pub fn init_fan_out(pool: rayon::ThreadPool) -> Result<(), rayon::ThreadPool> {
    FAN_OUT.set(pool)
}

/// The installed pool, or a default built on first use.
///
/// The default takes rayon's own width and exists for a caller with no `main` to install
/// one, such as the unit tests.
fn fan_out() -> &'static rayon::ThreadPool {
    FAN_OUT.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("hushar-fanout-{i}"))
            .build()
            .expect("a default rayon pool is buildable")
    })
}

/// Scores a request's rows, cutting them into mini batches when `mini_batch` is set.
///
/// Rows come back in request order however the batches interleave. An empty request is
/// answered without calling the engine. A request's features are read from `columns` and
/// left there, so the caller still owns them afterwards for the inference log. They reach
/// the log as they were received, before any transformation, and the whole feature map is
/// kept including keys the configuration ignores.
///
/// # Arguments
///
/// * `session` - the engine every mini batch of this request runs on
/// * `columns` - the request's rows, one column per feature
/// * `builder` - the inputs, resolved against the model's signature at startup
/// * `plan` - how to cut the rows, settled at startup
/// * `logging` - whether the features are kept for the inference log
///
/// # Returns
///
/// The batch's scores, row-major, and how long each stage took. When mini batches
/// overlap, a stage is reported as the longest single batch's rather than the sum of all
/// of them.
///
/// # Errors
///
/// Fails if the inputs cannot be built from the request, if the engine fails, or if a
/// mini batch panics -- a panic is turned into an error rather than unwinding into the
/// pool.
pub fn score_features(
    session: &dyn InferenceBackend,
    columns: &RequestColumns,
    builder: &InputBuilder,
    plan: &ExecutionPlan,
    logging: FeatureLogging,
) -> Result<ScoredBatch, crate::inference::InferenceError> {
    let total = columns.rows();
    if total == 0 {
        return Ok(ScoredBatch::empty());
    }
    let (size, pad_to) = match plan.mini_batch() {
        Some(m) => (m.size.max(1), m.is_fixed.then(|| m.size.max(1))),
        None => (total, None),
    };

    let mut mini_batches: Vec<Range<usize>> = Vec::with_capacity(total.div_ceil(size));
    let mut at = 0;
    while at < total {
        let end = (at + size).min(total);
        mini_batches.push(at..end);
        at = end;
    }

    if mini_batches.len() == 1 {
        let mut scored = empty_scored(total, logging);
        let rows = mini_batches.pop().unwrap_or(0..total);
        score_mini_batch(
            session,
            columns,
            rows,
            builder,
            pad_to,
            &mut scored,
            logging,
        )?;
        scored.set_fill(total, total + scored.timings.ballast_rows as usize);
        return Ok(scored);
    }

    let at_once = session.profile().concurrent_mini_batches();

    let scores_one = |rows: Range<usize>| -> Result<ScoredBatch, crate::inference::InferenceError> {
        let mut part = empty_scored(rows.end - rows.start, logging);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            score_mini_batch(
                session,
                columns,
                rows.clone(),
                builder,
                pad_to,
                &mut part,
                logging,
            )
        })) {
            Ok(Ok(())) => Ok(part),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(crate::inference::InferenceError::from(
                "a mini batch panicked while scoring",
            )),
        }
    };

    // Mini batches are dispatched in rounds no wider than the engine can actually
    // overlap. Spawning more than that would not run more at once; it would only queue
    // them inside the engine, where the wait is invisible. The calling thread takes the
    // first of each round, so a round of `n` occupies `n - 1` fan-out threads.
    let mut parts: Vec<Result<ScoredBatch, crate::inference::InferenceError>> =
        Vec::with_capacity(mini_batches.len());
    for round in mini_batches.chunks(at_once) {
        let (first, rest) = round.split_first().expect("a round is never empty");
        let mut spawned: Vec<Option<Result<ScoredBatch, crate::inference::InferenceError>>> =
            (0..rest.len()).map(|_| None).collect();
        let own = fan_out().in_place_scope(|scope| {
            for (rows, slot) in rest.iter().cloned().zip(spawned.iter_mut()) {
                scope.spawn(move |_| *slot = Some(scores_one(rows)));
            }
            scores_one(first.clone())
        });
        // The round's own mini batch is its first, so pushing it before the spawned ones
        // keeps `parts` in request order without sorting anything.
        parts.push(own);
        for slot in spawned {
            parts.push(slot.unwrap_or_else(|| {
                Err(crate::inference::InferenceError::from(
                    "a mini batch never reported a result",
                ))
            }));
        }
    }

    let mut scored = empty_scored(total, logging);
    for part in parts {
        let part = part?;
        append_scores(&mut scored.scores, part.scores)?;
        // A stage is the longest single mini batch's, since they overlap. The counts are
        // the request's own totals, so they add.
        scored.timings.vec_time = scored.timings.vec_time.max(part.timings.vec_time);
        scored.timings.tensor_time = scored.timings.tensor_time.max(part.timings.tensor_time);
        scored.timings.inference_time = scored
            .timings
            .inference_time
            .max(part.timings.inference_time);
        scored.timings.engine_calls += part.timings.engine_calls;
        scored.timings.ballast_rows += part.timings.ballast_rows;
    }
    scored.set_fill(total, total + scored.timings.ballast_rows as usize);

    Ok(scored)
}

/// An empty result sized for the rows it is about to hold.
///
/// The score precision is settled by the first batch appended, so an empty request has
/// none.
fn empty_scored(rows: usize, _logging: FeatureLogging) -> ScoredBatch {
    let _ = rows;
    ScoredBatch {
        scores: ScoreData::F32(Vec::new()),
        timings: RequestReport::default(),
    }
}

/// Scores one mini batch, padding it up to `pad_to` first when that is set, and appends
/// the real rows' results to `scored`.
///
/// Padding is added at the end of each input buffer rather than by appending rows, so no
/// synthetic row is built and none can reach the response or the log. `pad_to` is `None`
/// for a model whose row axis is dynamic.
#[allow(clippy::too_many_arguments)]
fn score_mini_batch(
    backend: &dyn InferenceBackend,
    columns: &RequestColumns,
    rows: Range<usize>,
    builder: &InputBuilder,
    pad_to: Option<usize>,
    scored: &mut ScoredBatch,
    logging: FeatureLogging,
) -> Result<(), crate::inference::InferenceError> {
    let vec_start = Instant::now();

    let real = rows.end - rows.start;
    let sent = pad_to.filter(|&size| real < size).unwrap_or(real);

    let inputs = builder.build_padded(columns, rows.clone(), sent)?;
    let build_start = Instant::now();

    run_batch(
        backend,
        &columns.row_ids()[rows],
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
/// Only the model's first output is returned -- class probabilities, a regression value,
/// a flattened forecast horizon or an embedding -- each one numeric vector per row.
///
/// # Arguments
///
/// * `backend` - the engine to run
/// * `kept` - the mini batch's slice of the request's row ids, cloned rather than moved
///   because the columns they came from are shared across mini batches
/// * `inputs` - the built inputs, already padded to `sent` rows
/// * `sent` - how many rows the engine was given, which is `kept.len()` plus any padding.
///   Only the first `kept.len()` rows are read back, so a padded row is scored and
///   dropped.
/// * `stage_start` - when building the inputs began; `stage_start` to `build_start` is
///   charged to the vectorization stage
/// * `build_start` - when the inputs were ready; `build_start` to the engine call is
///   charged to the tensor stage, and the call itself to the inference stage
/// * `scored` - the accumulator the rows' scores are appended to and the three stage
///   timings are added onto
/// * `logging` - whether this request's features are kept for the inference log
///
/// # Errors
///
/// Fails if the model returns no outputs, if it does not return one row per row sent, or
/// if its primary output is shorter than the rows kept.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    backend: &dyn InferenceBackend,
    kept: &[String],
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

    let width = primary.width();
    let keep = kept.len() * width;
    let taken = match &primary.data {
        ScoreData::F32(v) => ScoreData::F32(
            v.get(..keep)
                .ok_or_else(|| short_output(primary, keep, v.len()))?
                .to_vec(),
        ),
        ScoreData::F64(v) => ScoreData::F64(
            v.get(..keep)
                .ok_or_else(|| short_output(primary, keep, v.len()))?
                .to_vec(),
        ),
    };
    append_scores(&mut scored.scores, taken)?;
    let _ = logging;

    scored.timings.vec_time += (build_start - stage_start).as_micros();
    scored.timings.tensor_time += (inference_start - build_start).as_micros();
    scored.timings.inference_time += (inference_end - inference_start).as_micros();
    scored.timings.engine_calls += 1;
    scored.timings.ballast_rows += (sent - kept.len()) as u128;

    Ok(())
}

/// A whole batch's scores in the model's own precision, as a [`ScoreType`].
///
/// `FP64` is kept as doubles rather than narrowed. Infallible: scores are `FP32` or
/// `FP64` by construction.
pub fn to_score_type(scores: ScoreData) -> ScoreType {
    ScoreType {
        score_type: Some(match scores {
            ScoreData::F32(values) => Scores::FloatScores(FloatArray { values }),
            ScoreData::F64(values) => Scores::DoubleScores(DoubleArray { values }),
        }),
    }
}

/// Appends one batch's scores onto the accumulator, in one precision.
///
/// An empty accumulator has no precision yet, so the first batch settles it.
///
/// # Errors
///
/// Fails if the accumulator and the batch disagree about precision, which would otherwise
/// silently drop half a response.
pub(crate) fn append_scores(
    into: &mut ScoreData,
    from: ScoreData,
) -> Result<(), crate::inference::InferenceError> {
    match (into, from) {
        (ScoreData::F32(acc), ScoreData::F32(more)) => acc.extend(more),
        (ScoreData::F64(acc), ScoreData::F64(more)) => acc.extend(more),
        (acc @ ScoreData::F32(_), more) if acc.len() == 0 => *acc = more,
        (acc, more) => {
            return Err(format!(
                "mini batches disagree about precision: {} then {}",
                acc.element_type(),
                more.element_type()
            )
            .into());
        }
    }
    Ok(())
}

/// An output shorter than the rows it was asked to score.
fn short_output(primary: &Output, wanted: usize, got: usize) -> crate::inference::InferenceError {
    format!(
        "model output {:?} returned {got} values, but {wanted} were needed for the rows kept",
        primary.name
    )
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::vectorization_config::VectorizationConfig;
    use crate::inference::backend::test_support::{
        DoubleBackend, FailingBackend, MultiOutputBackend, RecordingBackend, SlowBackend,
        SummingBackend,
    };
    use crate::inference::input_builder::test_support::{V, request, row};
    use std::sync::Arc;
    use std::time::Duration;

    /// A builder for `backend`, from the given configuration JSON.
    fn builder_for(backend: &dyn InferenceBackend, json: &str) -> InputBuilder {
        let config = VectorizationConfig::from_json(json).expect("valid configuration");
        InputBuilder::resolve(config, backend.inputs()).expect("fits the model")
    }

    /// A vectorised configuration of `width` pass-through features named `f0..`.
    fn vector_config(width: usize, data_type: &str) -> String {
        let names: Vec<String> = (0..width).map(|i| format!("\"f{i}\"")).collect();
        format!(
            r#"{{"data_type": "{data_type}", "feature_order": [{}]}}"#,
            names.join(", ")
        )
    }

    /// A configuration mixing a numeric feature with a text one, three values wide.
    ///
    /// The text feature is looked up in a table, so it reaches the model as floats while
    /// the request keeps the text it sent.
    fn text_and_number_config() -> &'static str {
        r#"{
            "feature_transformations": {
                "age": {"type": "identity", "default_val": [0.0]},
                "tags": {
                    "type": "embedding_lookup",
                    "embeddings": {"london": [1.0, 2.0]},
                    "default_val": [0.0, 0.0]
                }
            },
            "feature_order": ["age", "tags"]
        }"#
    }

    /// A request of `rows` rows whose features `f0..fn` all hold `value`.
    fn numeric_request(rows: usize, width: usize, value: f32) -> RequestColumns {
        let ids: Vec<String> = (0..rows).map(|r| format!("row-{r}")).collect();
        let names: Vec<String> = (0..width).map(|i| format!("f{i}")).collect();
        let features: Vec<(&str, V)> = names
            .iter()
            .map(|name| (name.as_str(), V::f32(value)))
            .collect();
        let built: Vec<_> = ids.iter().map(|id| row(id, &features)).collect();
        request(&built)
    }

    /// The features an inference log batch carries: the request's own map, given back.
    type LoggedFeatures = std::collections::HashMap<String, hushar::hushar_proto::Feature>;

    /// The text one feature carries in the map handed to the inference log.
    ///
    /// The log carries the request's own features, so this reads them back the way an
    /// offline consumer would.
    fn logged_text(features: &LoggedFeatures, rows: usize, name: &str) -> Vec<String> {
        let columns = crate::inference::proto_columns::decode(features.clone(), rows)
            .expect("the log carries what the request sent");
        match columns.get(name) {
            Some(crate::inference::batch::FeatureData::Str(values)) => values.clone(),
            other => panic!("expected text for feature {name:?}, got {other:?}"),
        }
    }

    /// How many features the logged map carries.
    fn logged_count(features: &LoggedFeatures, rows: usize) -> usize {
        crate::inference::proto_columns::decode(features.clone(), rows)
            .expect("the log carries what the request sent")
            .len()
    }

    /// Row `r`'s slice of the batch's one flat buffer, of `rows` rows.
    ///
    /// The buffer is row-major, so the width is `len / rows` and row `r` is
    /// `r * width .. (r + 1) * width`.
    fn float_row(scores: &ScoreData, rows: usize, r: usize) -> Vec<f32> {
        match scores {
            ScoreData::F32(v) => {
                let width = v.len() / rows.max(1);
                v[r * width..(r + 1) * width].to_vec()
            }
            other => panic!("expected float scores, got {other:?}"),
        }
    }

    /// The same, for a model kept at double precision.
    fn double_row(scores: &ScoreData, rows: usize, r: usize) -> Vec<f64> {
        match scores {
            ScoreData::F64(v) => {
                let width = v.len() / rows.max(1);
                v[r * width..(r + 1) * width].to_vec()
            }
            other => panic!("expected double scores, got {other:?}"),
        }
    }

    /// Every row is scored, and the scores land back to back in one buffer.
    #[test]
    fn the_feature_path_scores_every_row() {
        let backend = SummingBackend::new(3, 2);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &numeric_request(4, 3, 1.0),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(
            scored.scores.len(),
            8,
            "four rows of two scores, back to back in one buffer"
        );
    }

    /// A slow engine call is reported against the inference stage, not vectorisation.
    #[test]
    fn inference_time_measures_the_engine_call() {
        let delay = Duration::from_millis(5);
        let backend = SlowBackend::new(3, 2, delay);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &numeric_request(4, 3, 1.0),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("scored");

        let floor = delay.as_micros() / 2;
        assert!(
            scored.timings.inference_time >= floor,
            "a {delay:?} engine call should report at least {floor}us of inference time, \
             got {}us",
            scored.timings.inference_time,
        );
        assert!(
            scored.timings.vec_time < scored.timings.inference_time,
            "the delay belongs to the engine call, but vectorisation reported {}us \
             against the engine's {}us",
            scored.timings.vec_time,
            scored.timings.inference_time,
        );
    }

    /// Each row is scored from its own values, in request order.
    #[test]
    fn each_row_receives_its_own_scores() {
        let backend = SummingBackend::new(2, 1);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let columns = request(&[
            row("a", &[("f0", V::f32(1.0)), ("f1", V::f32(1.0))]),
            row("b", &[("f0", V::f32(5.0)), ("f1", V::f32(5.0))]),
        ]);
        let scored = score_features(&backend, &columns, &builder, &WHOLE, FeatureLogging::Record)
            .expect("scored");

        assert_eq!(columns.row_ids(), ["a".to_owned(), "b".to_owned()]);
        assert_eq!(float_row(&scored.scores, 2, 0), vec![2.0]);
        assert_eq!(float_row(&scored.scores, 2, 1), vec![10.0]);
    }

    /// The columns handed to the log hold every feature sent, ignored keys included.
    #[test]
    fn the_columns_handed_to_the_log_keep_every_feature_sent() {
        let backend = SummingBackend::new(1, 1);
        let builder = builder_for(&backend, &vector_config(1, "float"));

        let columns = request(&[row(
            "a",
            &[("f0", V::f32(2.0)), ("ignored", V::str("kept anyway"))],
        )]);
        let scored = score_features(&backend, &columns, &builder, &WHOLE, FeatureLogging::Record)
            .expect("scored");

        assert_eq!(scored.scores.as_f32(), vec![2.0]);

        let (row_ids, features) = columns.into_wire();
        assert_eq!(
            row_ids,
            vec!["a".to_owned()],
            "the row ids reach the log as their own column, beside the features"
        );
        assert_eq!(
            logged_count(&features, 1),
            2,
            "both configured keys survive into the log, the ignored one included"
        );
        assert_eq!(
            logged_text(&features, 1, "ignored"),
            vec!["kept anyway".to_owned()]
        );
    }

    /// An empty request returns without calling the engine.
    #[test]
    fn an_empty_batch_is_not_sent_to_the_engine() {
        let backend = FailingBackend;
        let builder = builder_for(&backend, &vector_config(1, "float"));

        let scored = score_features(
            &backend,
            &request(&[]),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("nothing to do");
        assert_eq!(scored.scores.len(), 0);
        assert_eq!(scored.timings.inference_time, 0);
    }

    /// `Skip` is a logging lever only: every row is still scored.
    #[test]
    fn a_skipped_request_is_still_scored() {
        let backend = SummingBackend::new(2, 1);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let scored = score_features(
            &backend,
            &numeric_request(3, 2, 2.0),
            &builder,
            &WHOLE,
            FeatureLogging::Skip,
        )
        .expect("scored");

        assert_eq!(scored.scores.len(), 3, "every row is still answered");
        assert_eq!(float_row(&scored.scores, 3, 0), vec![4.0]);
    }

    /// The logging decision does not change the scores a caller receives.
    #[test]
    fn skipping_the_log_does_not_change_the_response() {
        let recorded = {
            let backend = SummingBackend::new(2, 1);
            let builder = builder_for(&backend, &vector_config(2, "float"));
            score_features(
                &backend,
                &numeric_request(4, 2, 1.5),
                &builder,
                &fixed(3),
                FeatureLogging::Record,
            )
            .expect("scored")
        };
        let skipped = {
            let backend = SummingBackend::new(2, 1);
            let builder = builder_for(&backend, &vector_config(2, "float"));
            score_features(
                &backend,
                &numeric_request(4, 2, 1.5),
                &builder,
                &fixed(3),
                FeatureLogging::Skip,
            )
            .expect("scored")
        };

        assert_eq!(recorded.scores, skipped.scores);
        assert_eq!(recorded.scores.len(), 4);
    }

    /// A text feature reaches the log as the text it was, whatever it became on the way
    /// in.
    ///
    /// Both a value the table knows and a long one it does not, so the log is shown to
    /// keep the feature intact rather than the floats it produced.
    #[test]
    fn a_text_feature_is_logged_as_the_text_it_was() {
        let backend = SummingBackend::new(3, 1);
        let builder = builder_for(&backend, text_and_number_config());

        let long_text = "x".repeat(4096);
        let columns = request(&[
            row("a", &[("age", V::f32(1.0)), ("tags", V::str("london"))]),
            row("b", &[("age", V::f32(2.0)), ("tags", V::str(&long_text))]),
        ]);
        score_features(&backend, &columns, &builder, &WHOLE, FeatureLogging::Record)
            .expect("scored");

        let (_, features) = columns.into_wire();
        assert_eq!(
            logged_text(&features, 2, "tags"),
            vec!["london".to_owned(), long_text],
            "the log keeps text as text, and a long value arrives whole"
        );
    }

    /// An engine failure is returned to the caller.
    #[test]
    fn an_engine_failure_is_propagated() {
        let backend = FailingBackend;
        let builder = builder_for(&backend, &vector_config(1, "float"));

        let err = score_features(
            &backend,
            &numeric_request(1, 1, 1.0),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect_err("the engine failed");
        assert!(err.to_string().contains("engine exploded"), "got {err}");
    }

    /// A single-precision model's scores come back as floats.
    #[test]
    fn a_single_precision_model_returns_float_scores() {
        let backend = SummingBackend::new(2, 1);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let scored = score_features(
            &backend,
            &numeric_request(1, 2, 1.5),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("scored");
        assert!(matches!(
            to_score_type(scored.scores).score_type,
            Some(Scores::FloatScores(_))
        ));
    }

    /// A double-precision model's scores are narrowed neither on the way in nor out; 0.1
    /// has no exact `f32`.
    #[test]
    fn a_double_precision_model_returns_double_scores_unnarrowed() {
        let backend = DoubleBackend::new(1);
        let builder = builder_for(&backend, &vector_config(1, "double"));

        let columns = request(&[row("a", &[("f0", V::f64(0.1))])]);
        let scored = score_features(&backend, &columns, &builder, &WHOLE, FeatureLogging::Record)
            .expect("scored");
        assert_eq!(double_row(&scored.scores, 1, 0), vec![0.1]);
        assert!(matches!(
            to_score_type(scored.scores).score_type,
            Some(Scores::DoubleScores(_))
        ));
    }

    /// Double scores are read from the row they belong to.
    #[test]
    fn double_scores_are_read_from_the_right_row() {
        let backend = DoubleBackend::new(1);
        let builder = builder_for(&backend, &vector_config(1, "double"));

        let columns = request(&[
            row("a", &[("f0", V::f64(0.25))]),
            row("b", &[("f0", V::f64(0.75))]),
        ]);
        let scored = score_features(&backend, &columns, &builder, &WHOLE, FeatureLogging::Record)
            .expect("scored");
        assert_eq!(double_row(&scored.scores, 2, 0), vec![0.25]);
        assert_eq!(double_row(&scored.scores, 2, 1), vec![0.75]);
    }

    /// Only the model's first output reaches the score buffer.
    #[test]
    fn only_the_models_first_output_is_returned() {
        let backend = MultiOutputBackend::new(2);
        let builder = builder_for(&backend, &vector_config(2, "float"));

        let scored = score_features(
            &backend,
            &numeric_request(2, 2, 1.5),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(float_row(&scored.scores, 2, 0), vec![3.0, -3.0]);
        assert_eq!(scored.scores.as_f32(), vec![3.0, -3.0, 3.0, -3.0]);
    }

    /// A padded split on one shared session, which is what a pinned graph needs.
    fn fixed(size: usize) -> ExecutionPlan {
        ExecutionPlan::SharedSession {
            mini_batch: MiniBatch {
                size,
                is_fixed: true,
            },
        }
    }

    /// Every request in one call, which is the default plan.
    const WHOLE: ExecutionPlan = ExecutionPlan::WholeRequest;

    /// A backend that records how many calls overlap, and the widest overlap it saw.
    ///
    /// Counts entries and exits around a sleep long enough that two genuinely concurrent
    /// calls must be inside it at the same time.
    #[derive(Debug)]
    struct OverlapWatcher {
        inner: SummingBackend,
        in_flight: std::sync::Mutex<usize>,
        widest: std::sync::Mutex<usize>,
    }

    impl OverlapWatcher {
        fn new(profile: crate::inference::backend::ExecutionProfile) -> Self {
            Self {
                inner: SummingBackend::new(3, 1).with_profile(profile),
                in_flight: std::sync::Mutex::new(0),
                widest: std::sync::Mutex::new(0),
            }
        }

        /// The most calls that were ever inside `run` together.
        fn widest(&self) -> usize {
            *self.widest.lock().expect("no test panics holding this")
        }
    }

    impl InferenceBackend for OverlapWatcher {
        fn name(&self) -> &str {
            "test/overlap-watcher"
        }
        fn profile(&self) -> crate::inference::backend::ExecutionProfile {
            self.inner.profile()
        }
        fn required_rows(&self) -> Option<usize> {
            self.inner.required_rows()
        }
        fn inputs(&self) -> &[crate::inference::batch::IoSpec] {
            self.inner.inputs()
        }
        fn outputs(&self) -> &[crate::inference::batch::IoSpec] {
            self.inner.outputs()
        }
        fn run(
            &self,
            inputs: InputBatch,
        ) -> Result<crate::inference::batch::OutputBatch, crate::inference::InferenceError>
        {
            {
                let mut in_flight = self.in_flight.lock().expect("no panic");
                *in_flight += 1;
                let mut widest = self.widest.lock().expect("no panic");
                *widest = (*widest).max(*in_flight);
            }
            std::thread::sleep(Duration::from_millis(30));
            *self.in_flight.lock().expect("no panic") -= 1;
            self.inner.run(inputs)
        }
    }

    /// Asserts a backend that overlaps nothing never has two mini batches inside `run`
    /// at the same time.
    ///
    /// This is the case a pinned graph on a device provider reaches: the mini batch is
    /// the model's own call shape, so it cannot be refused, and running the calls
    /// concurrently would only queue them where the wait cannot be seen.
    #[test]
    fn a_profile_of_one_never_runs_two_mini_batches_at_once() {
        let backend =
            OverlapWatcher::new(crate::inference::backend::ExecutionProfile::serialized());
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(8, 3),
            &builder,
            &dynamic(1),
            FeatureLogging::Record,
        )
        .expect("eight single-row mini batches");

        assert_eq!(scored.scores.len(), 8, "every row is still scored");
        assert_eq!(
            backend.widest(),
            1,
            "a serializing backend must run its mini batches one at a time"
        );
    }

    /// Asserts a backend that overlaps does run its mini batches at the same time, so the
    /// cap bounds concurrency without removing it.
    #[test]
    fn an_overlapping_profile_runs_mini_batches_together_up_to_its_width() {
        let backend = OverlapWatcher::new(crate::inference::backend::ExecutionProfile::new(2));
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(8, 3),
            &builder,
            &dynamic(1),
            FeatureLogging::Record,
        )
        .expect("eight single-row mini batches");

        assert_eq!(scored.scores.len(), 8);
        assert_eq!(
            backend.widest(),
            2,
            "two is what this backend declared it overlaps, so two is what should run"
        );
    }

    /// Asserts the plan decides, not the row count: the same request and backend give
    /// one call under `WholeRequest` and several under a mini batch, with equal scores.
    #[test]
    fn the_plan_decides_how_many_calls_a_request_becomes() {
        let whole_backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&whole_backend, &vector_config(3, "float"));
        let whole = score_features(
            &whole_backend,
            &ramp_request(6, 3),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("one call");
        assert_eq!(whole_backend.batches(), vec![6], "one call of every row");

        let split_backend = RecordingBackend::new(3, 1);
        let split = score_features(
            &split_backend,
            &ramp_request(6, 3),
            &builder,
            &dynamic(2),
            FeatureLogging::Record,
        )
        .expect("three calls");
        let mut sizes = split_backend.batches();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![2, 2, 2], "three calls of two rows");

        assert_eq!(
            whole.scores.as_f32(),
            split.scores.as_f32(),
            "the plan changes the calls, never the scores"
        );
    }

    /// A split on one shared session whose last batch is however many rows remain.
    fn dynamic(size: usize) -> ExecutionPlan {
        ExecutionPlan::SharedSession {
            mini_batch: MiniBatch {
                size,
                is_fixed: false,
            },
        }
    }

    /// A request of `rows` rows of `width` features, where row `r` holds `r` in every
    /// feature.
    ///
    /// A row's sum is therefore `width * r`, so its scores name the row they came from.
    fn ramp_request(rows: usize, width: usize) -> RequestColumns {
        let ids: Vec<String> = (0..rows).map(|r| format!("row-{r}")).collect();
        let names: Vec<String> = (0..width).map(|i| format!("f{i}")).collect();
        let built: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(r, id)| {
                let features: Vec<(&str, V)> = names
                    .iter()
                    .map(|name| (name.as_str(), V::f32(r as f32)))
                    .collect();
                row(id, &features)
            })
            .collect();
        request(&built)
    }

    /// A request larger than the fixed size is split into batches of it, in row order.
    #[test]
    fn a_request_larger_than_the_fixed_size_is_split_into_batches_of_it() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(6, 3),
            &builder,
            &fixed(4),
            FeatureLogging::Record,
        )
        .expect("six rows against a size of four");

        assert_eq!(backend.batches(), vec![4, 4]);
        assert_eq!(scored.scores.len(), 6);
        for r in 0..6 {
            assert_eq!(float_row(&scored.scores, 6, r), vec![3.0 * r as f32]);
        }
    }

    /// A request smaller than the fixed size is padded up to it, and the ballast dropped.
    #[test]
    fn a_request_smaller_than_the_fixed_size_is_padded_up_to_it() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(1, 3),
            &builder,
            &fixed(4),
            FeatureLogging::Record,
        )
        .expect("one row against a size of four");

        assert_eq!(backend.batches(), vec![4]);
        assert_eq!(scored.scores.len(), 1);
    }

    /// An exact multiple of the fixed size is split evenly, with nothing padded.
    #[test]
    fn an_exact_multiple_of_the_fixed_size_is_split_evenly() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(8, 3),
            &builder,
            &fixed(4),
            FeatureLogging::Record,
        )
        .expect("eight rows against a size of four");

        assert_eq!(backend.batches(), vec![4, 4]);
        assert_eq!(scored.scores.len(), 8);
        assert_eq!(float_row(&scored.scores, 8, 7), vec![21.0]);
    }

    /// Padding a configuration with a text feature uses its transformation's default, so
    /// no ballast score reaches the caller.
    #[test]
    fn a_fixed_size_pads_a_configuration_with_a_text_feature() {
        let backend = SummingBackend::new(3, 1);
        let builder = builder_for(&backend, text_and_number_config());

        let columns = request(&[
            row("a", &[("age", V::f32(1.0)), ("tags", V::str("x"))]),
            row("b", &[("age", V::f32(2.0)), ("tags", V::str("y"))]),
            row("c", &[("age", V::f32(3.0)), ("tags", V::str("london"))]),
        ]);
        let scored = score_features(
            &backend,
            &columns,
            &builder,
            &fixed(4),
            FeatureLogging::Record,
        )
        .expect("padding must not require features the caller never sent");

        assert_eq!(scored.scores.len(), 3, "only the real rows come back");
        assert_eq!(
            scored.scores.as_f32(),
            vec![1.0, 2.0, 6.0],
            "no ballast score reaches the caller"
        );
    }

    /// No padded row's score reaches the response or the log.
    #[test]
    fn padding_reaches_neither_the_response_nor_the_log() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(5, 3),
            &builder,
            &fixed(4),
            FeatureLogging::Record,
        )
        .expect("five rows against a size of four");

        assert_eq!(backend.batches(), vec![4, 4]);
        assert_eq!(scored.scores.len(), 5);
        assert_eq!(
            scored.scores.as_f32(),
            vec![0.0, 3.0, 6.0, 9.0, 12.0],
            "the three ballast rows scored 0 and none of them survived"
        );
    }

    /// A fixed size of one runs one batch per row.
    #[test]
    fn a_fixed_size_of_one_runs_a_batch_per_row() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(3, 3),
            &builder,
            &fixed(1),
            FeatureLogging::Record,
        )
        .expect("three rows against a size of one");

        assert_eq!(backend.batches(), vec![1, 1, 1]);
        assert_eq!(scored.scores.len(), 3);
        assert_eq!(float_row(&scored.scores, 3, 2), vec![6.0]);
    }

    /// A dynamic split leaves the last batch short rather than padding it. Batch sizes
    /// are sorted before comparing, because the batches finish in whatever order the
    /// scheduler gives them.
    #[test]
    fn a_dynamic_split_leaves_the_last_batch_short() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(6, 3),
            &builder,
            &dynamic(4),
            FeatureLogging::Record,
        )
        .expect("six rows split into four and two");

        let mut sizes = backend.batches();
        sizes.sort_unstable();
        assert_eq!(
            sizes,
            vec![2, 4],
            "the last batch carries the two rows that remain, unpadded"
        );
        assert_eq!(
            scored.scores.len(),
            6,
            "no padding row reaches the response or the log, because none was made"
        );
    }

    /// Rows stay in request order when the batches finish in the reverse of the order
    /// they were submitted.
    #[test]
    fn request_order_survives_batches_finishing_out_of_order() {
        const ROWS: usize = 16;

        #[derive(Debug)]
        struct ReverseDelay {
            inner: SummingBackend,
        }

        impl InferenceBackend for ReverseDelay {
            fn name(&self) -> &str {
                "test/reverse-delay"
            }
            fn profile(&self) -> crate::inference::backend::ExecutionProfile {
                self.inner.profile()
            }
            fn required_rows(&self) -> Option<usize> {
                self.inner.required_rows()
            }
            fn inputs(&self) -> &[crate::inference::batch::IoSpec] {
                self.inner.inputs()
            }
            fn outputs(&self) -> &[crate::inference::batch::IoSpec] {
                self.inner.outputs()
            }
            fn run(
                &self,
                inputs: InputBatch,
            ) -> Result<crate::inference::batch::OutputBatch, crate::inference::InferenceError>
            {
                let sum: f32 = inputs
                    .get(self.inner.input_name())
                    .and_then(|c| c.row_as_f32(0).map(|r| r.iter().sum()))
                    .unwrap_or_default();
                let steps = (3 * (ROWS - 1)) as f32 - sum;
                std::thread::sleep(Duration::from_millis(2 * steps.max(0.0) as u64));
                self.inner.run(inputs)
            }
        }

        let backend = ReverseDelay {
            inner: SummingBackend::new(3, 1),
        };
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(ROWS, 3),
            &builder,
            &dynamic(1),
            FeatureLogging::Record,
        )
        .expect("sixteen single-row batches");

        assert_eq!(scored.scores.len(), ROWS);
        for r in 0..ROWS {
            assert_eq!(
                float_row(&scored.scores, ROWS, r),
                vec![3.0 * r as f32],
                "row {r} is out of place, so the fan-out collected on completion"
            );
        }
        let ramp: Vec<f32> = (0..ROWS).map(|r| 3.0 * r as f32).collect();
        assert_eq!(scored.scores.as_f32(), ramp);
    }

    /// Twenty single-row batches come back in request order.
    #[test]
    fn mini_batches_return_rows_in_request_order() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(20, 3),
            &builder,
            &dynamic(1),
            FeatureLogging::Record,
        )
        .expect("twenty single-row batches");

        assert_eq!(backend.batches().len(), 20);
        assert_eq!(scored.scores.len(), 20);
        for r in 0..20 {
            assert_eq!(
                float_row(&scored.scores, 20, r),
                vec![3.0 * r as f32],
                "row {r} came back out of request order"
            );
        }
    }

    /// Splitting the rows changes when they are scored, not what they score.
    #[test]
    fn a_dynamic_split_does_not_change_a_rows_scores() {
        let whole = RecordingBackend::new(3, 1);
        let split = RecordingBackend::new(3, 1);
        let builder = builder_for(&whole, &vector_config(3, "float"));

        let one = score_features(
            &whole,
            &ramp_request(7, 3),
            &builder,
            &WHOLE,
            FeatureLogging::Skip,
        )
        .expect("scored whole");
        let many = score_features(
            &split,
            &ramp_request(7, 3),
            &builder,
            &dynamic(3),
            FeatureLogging::Skip,
        )
        .expect("scored in mini batches");

        assert_eq!(whole.batches(), vec![7]);
        let mut sizes = split.batches();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 3, 3]);
        assert_eq!(one.scores.len(), many.scores.len());
        for r in 0..7 {
            assert_eq!(
                float_row(&one.scores, 7, r),
                float_row(&many.scores, 7, r),
                "row {r} scored differently when split"
            );
        }
        assert_eq!(one.scores, many.scores);
    }

    /// With no mini batch declared, the whole request runs as one batch.
    #[test]
    fn a_dynamic_deployment_runs_the_whole_request_as_one_batch() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(6, 3),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(backend.batches(), vec![6]);
        assert_eq!(scored.scores.len(), 6);
    }

    /// An empty request runs no batch even when a fixed size is set.
    #[test]
    fn an_empty_request_runs_no_batch_even_when_pinned() {
        let backend = RecordingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &request(&[]),
            &builder,
            &fixed(4),
            FeatureLogging::Record,
        )
        .expect("nothing");

        assert!(backend.batches().is_empty());
        assert_eq!(scored.scores.len(), 0);
    }

    /// A row's scores do not depend on what shared its batch: the same rows scored whole
    /// and in padded batches of two agree row for row.
    #[test]
    fn splitting_and_padding_do_not_change_a_rows_scores() {
        let dynamic = RecordingBackend::new(3, 1);
        let dynamic_builder = builder_for(&dynamic, &vector_config(3, "float"));
        let whole = score_features(
            &dynamic,
            &ramp_request(7, 3),
            &dynamic_builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("scored");

        let pinned = RecordingBackend::new(3, 1);
        let pinned_builder = builder_for(&pinned, &vector_config(3, "float"));
        let split = score_features(
            &pinned,
            &ramp_request(7, 3),
            &pinned_builder,
            &fixed(2),
            FeatureLogging::Record,
        )
        .expect("scored");

        assert_eq!(dynamic.batches(), vec![7]);
        assert_eq!(pinned.batches(), vec![2, 2, 2, 2]);

        assert_eq!(whole.scores.len(), split.scores.len());
        for r in 0..7 {
            assert_eq!(
                float_row(&whole.scores, 7, r),
                float_row(&split.scores, 7, r),
                "row {r} scored differently when split and padded"
            );
        }
        assert_eq!(whole.scores, split.scores);
    }

    /// A `ScoreType` copied for the log holds the response's own numbers, unnarrowed.
    #[test]
    fn the_log_and_the_response_carry_the_same_scores() {
        let doubles = to_score_type(ScoreData::F64(vec![0.1, 0.2]));
        let logged = doubles.clone();

        match (
            doubles.score_type.expect("set"),
            logged.score_type.expect("set"),
        ) {
            (Scores::DoubleScores(response), Scores::DoubleScores(log)) => {
                assert_eq!(response.values, vec![0.1, 0.2]);
                assert_eq!(
                    log.values, response.values,
                    "the log holds the response's own numbers"
                );
            }
            _ => panic!("a double-precision batch must reach both sides as doubles"),
        }

        assert!(matches!(
            to_score_type(ScoreData::F32(vec![0.5])).score_type,
            Some(Scores::FloatScores(_))
        ));
    }

    // ---- mini batches on one session -----------------------------------------------

    /// A backend recording when each of its calls ran, so overlap is asserted rather than
    /// inferred from a wall-clock total.
    #[derive(Debug)]
    struct WindowBackend {
        inner: SummingBackend,
        delay: Duration,
        windows: Arc<std::sync::Mutex<Vec<(Instant, Instant)>>>,
    }

    impl WindowBackend {
        fn new(
            at_once: usize,
            delay: Duration,
            windows: Arc<std::sync::Mutex<Vec<(Instant, Instant)>>>,
        ) -> Self {
            Self {
                inner: SummingBackend::new(3, 1)
                    .with_profile(crate::inference::backend::ExecutionProfile::new(at_once)),
                delay,
                windows,
            }
        }
    }

    impl InferenceBackend for WindowBackend {
        fn name(&self) -> &str {
            "test/window"
        }
        fn profile(&self) -> crate::inference::backend::ExecutionProfile {
            self.inner.profile()
        }
        fn required_rows(&self) -> Option<usize> {
            self.inner.required_rows()
        }
        fn inputs(&self) -> &[crate::inference::batch::IoSpec] {
            self.inner.inputs()
        }
        fn outputs(&self) -> &[crate::inference::batch::IoSpec] {
            self.inner.outputs()
        }
        fn run(
            &self,
            inputs: InputBatch,
        ) -> Result<crate::inference::batch::OutputBatch, crate::inference::InferenceError>
        {
            let started = Instant::now();
            std::thread::sleep(self.delay);
            let ended = Instant::now();
            self.windows
                .lock()
                .expect("no test panics holding this")
                .push((started, ended));
            self.inner.run(inputs)
        }
    }

    /// Asserts two mini batches of one request are really in flight together on one
    /// session, rather than merely spawned and then serialized.
    ///
    /// The engine sleeps, so two calls that overlap in wall-clock time can only have run
    /// at the same time. This is the property `concurrent_mini_batches` promises.
    #[test]
    fn mini_batches_on_one_session_overlap_in_time() {
        let windows = Arc::new(std::sync::Mutex::new(Vec::new()));
        let backend = WindowBackend::new(2, Duration::from_millis(60), Arc::clone(&windows));
        let builder = builder_for(&backend, &vector_config(3, "float"));

        score_features(
            &backend,
            &ramp_request(4, 3),
            &builder,
            &dynamic(2),
            FeatureLogging::Record,
        )
        .expect("two mini batches on one session");

        let recorded = windows.lock().expect("no panic").clone();
        assert_eq!(recorded.len(), 2, "two mini batches, two calls");
        let (first_start, first_end) = recorded[0];
        let (second_start, second_end) = recorded[1];
        assert!(
            first_start < second_end && second_start < first_end,
            "the two calls must be in flight together, each took {:?} and {:?}",
            first_end.duration_since(first_start),
            second_end.duration_since(second_start),
        );
    }

    /// Asserts a whole request is one call with nothing discarded, so the counts start
    /// from an honest baseline.
    #[test]
    fn a_whole_request_reports_one_call_and_no_ballast() {
        let backend = SummingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let scored = score_features(
            &backend,
            &ramp_request(5, 3),
            &builder,
            &WHOLE,
            FeatureLogging::Record,
        )
        .expect("one call");

        assert_eq!(scored.timings.engine_calls, 1, "one call per request");
        assert_eq!(
            scored.timings.ballast_rows, 0,
            "a dynamic axis takes the rows as they are"
        );
        assert_eq!(scored.timings.fill_percent, 100);
    }

    /// Asserts a split request counts every call it became, and a padded last mini batch
    /// is reported as ballast rather than hidden.
    #[test]
    fn a_split_request_counts_its_calls_and_its_padding() {
        let backend = SummingBackend::new(3, 1);
        let builder = builder_for(&backend, &vector_config(3, "float"));

        let dynamic_split = score_features(
            &backend,
            &ramp_request(5, 3),
            &builder,
            &dynamic(2),
            FeatureLogging::Record,
        )
        .expect("three calls");
        assert_eq!(
            dynamic_split.timings.engine_calls, 3,
            "five rows at two per call is three calls"
        );
        assert_eq!(
            dynamic_split.timings.ballast_rows, 0,
            "a short last mini batch is sent short, not padded"
        );

        let padded_split = score_features(
            &backend,
            &ramp_request(5, 3),
            &builder,
            &fixed(2),
            FeatureLogging::Record,
        )
        .expect("three calls, the last padded");
        assert_eq!(padded_split.timings.engine_calls, 3);
        assert_eq!(
            padded_split.timings.ballast_rows, 1,
            "the last mini batch carried one row and was padded to two"
        );
        assert_eq!(
            padded_split.timings.fill_percent, 83,
            "five real rows of the six sent"
        );
        assert_eq!(
            padded_split.scores.as_f32(),
            dynamic_split.scores.as_f32(),
            "and padding never changes a score"
        );
    }
}
