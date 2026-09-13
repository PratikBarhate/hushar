// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Where inference timings go.
//!
//! Today that means stderr or Amazon CloudWatch. As with [`BlobStore`](super::BlobStore),
//! the point of the trait is what it makes cheap later: **adding a monitoring
//! backend is one implementation**, and nothing above this module changes.
//!
//! Recording happens on the request path, not on a sidecar. It is a lock and a few
//! additions for stderr, and a lock and a push for CloudWatch; the publish itself is
//! spawned once a batch is full. See
//! [`accumulator`](super::accumulator) for why that is cheaper than a channel and how
//! memory stays bounded.
//!
//! **Timings are never sampled.** Unlike the inference log, they are already an
//! aggregate — three numbers per batch, whatever the batch's size — so sampling would
//! buy nothing and would bias the mean it exists to report.

use std::fmt::Debug;
use std::sync::Arc;

use crate::inference::InferenceMicros;
use crate::io::accumulator::Accumulator;

/// Somewhere to send per-batch timings.
///
/// Implementations are shared across request handlers behind an [`Arc`], so every
/// method takes `&self`.
#[tonic::async_trait]
pub(crate) trait MetricsSink: Debug + Send + Sync {
    /// Which backend this is, for the startup banner.
    fn backend(&self) -> &str;

    /// Scored batches accumulated before one send or one summary, after any clamping
    /// the API forces.
    fn batch_size(&self) -> usize;

    /// Records one batch's timings, on the request path.
    ///
    /// Synchronous and non-blocking: it accumulates, and on the batch that fills the
    /// buffer it spawns the publish. Nothing here awaits a network call, because one
    /// request in every `batch_size` would then carry a CloudWatch round trip.
    ///
    /// `model_id` is the model that produced them. It is a parameter rather than a
    /// property of the sink because one process may serve two models during a
    /// roll-out, and averaging their timings together would hide the very difference
    /// the roll-out exists to measure.
    ///
    /// # Panics
    ///
    /// If called outside a tokio runtime, since the publish is spawned.
    fn record(&self, model_id: &str, timings: &InferenceMicros);

    /// Pushes anything buffered and waits for the sends already in the air.
    ///
    /// Called once at shutdown. A sink that batches would otherwise lose what it was
    /// holding, which is exactly the window where the numbers matter most.
    async fn flush(&self);
}

/// The three timings a batch produces, named once.
///
/// Shared by the implementations so a metric cannot be called one thing in
/// CloudWatch and another on stderr.
const STAGES: [&str; 3] = ["VectorizationTime", "ToTensorTime", "InferenceTime"];

/// Data points `PutMetricData` accepts in one call.
///
/// The call is rejected whole when it is exceeded, so an unclamped buffer would lose
/// everything in it rather than some of it.
const CLOUDWATCH_MAX_DATUMS: usize = 1000;

/// The largest capacity that both fits one call and lands on a batch boundary.
///
/// 1000 is not a multiple of the three data points a batch contributes, and the
/// accumulator sends on "at least capacity" rather than "exactly capacity" — so a
/// capacity of 1000 is first reached at 1002, which is an over-long call that
/// CloudWatch rejects whole. At the default batch size that rejected **every**
/// publish, and the only symptom was a recurring error line.
///
/// Rounding down to 999 makes the buffer land exactly on the boundary, which also
/// makes [`CloudWatchMetrics::batch_size`] the number of batches that really triggers
/// a send rather than one less.
const CLOUDWATCH_MAX_BATCHED_DATUMS: usize =
    CLOUDWATCH_MAX_DATUMS - CLOUDWATCH_MAX_DATUMS % STAGES.len();

/// Aggregates timings and writes periodic summaries to stderr.
///
/// The default, because it needs no credentials and no network, which makes a
/// local run and a test observable with no infrastructure. It aggregates rather
/// than printing per batch: one line per request would be unreadable at any real
/// rate and would itself become the bottleneck.
///
/// This is the one sink with no accumulator, because it has nothing to accumulate:
/// running totals are a fixed six words whatever the batch size, so there is no
/// buffer to bound and nothing to spawn.
#[derive(Debug)]
pub(crate) struct StderrMetrics {
    every: usize,
    /// One aggregate per model, so a roll-out's two arms are summarised separately.
    /// A map rather than a pair because the sink does not know how many models the
    /// service loaded, and it should not have to.
    state: std::sync::Mutex<std::collections::BTreeMap<String, Aggregate>>,
}

/// Running totals between summaries.
#[derive(Debug, Default)]
struct Aggregate {
    batches: u64,
    totals: [u128; 3],
    max_inference: u128,
}

impl StderrMetrics {
    /// Summarises every `every` batches. Zero is treated as one.
    pub(crate) fn new(every: usize) -> Self {
        Self {
            every: every.max(1),
            state: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    /// Recovers the guard even if a previous holder panicked.
    ///
    /// Metrics are not worth propagating a poisoning panic into the service.
    fn lock(&self) -> std::sync::MutexGuard<'_, std::collections::BTreeMap<String, Aggregate>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Renders and clears one model's running totals.
    ///
    /// The model is named on every line, because two arms writing unlabelled lines to
    /// the same stream would be indistinguishable.
    fn emit(model_id: &str, state: &mut Aggregate) {
        if state.batches == 0 {
            return;
        }
        let n = u128::from(state.batches);
        let means: Vec<String> = STAGES
            .iter()
            .zip(state.totals.iter())
            .map(|(name, total)| format!("{name} {}", total / n))
            .collect();
        eprintln!(
            "metrics: {model_id} | {} batches | mean us: {} | max InferenceTime {}",
            state.batches,
            means.join(", "),
            state.max_inference
        );
        *state = Aggregate::default();
    }
}

#[tonic::async_trait]
impl MetricsSink for StderrMetrics {
    fn backend(&self) -> &str {
        "stderr"
    }

    fn batch_size(&self) -> usize {
        self.every
    }

    fn record(&self, model_id: &str, timings: &InferenceMicros) {
        let mut models = self.lock();
        let state = models.entry(model_id.to_owned()).or_default();
        state.batches += 1;
        state.totals[0] += timings.vec_time;
        state.totals[1] += timings.tensor_time;
        state.totals[2] += timings.inference_time;
        state.max_inference = state.max_inference.max(timings.inference_time);
        // Counted per model, so each arm's line covers the same number of batches. A
        // shared counter would make the minority arm's line an average over a handful.
        if state.batches as usize >= self.every {
            Self::emit(model_id, state);
        }
    }

    async fn flush(&self) {
        let mut models = self.lock();
        for (model_id, state) in models.iter_mut() {
            Self::emit(model_id, state);
        }
    }
}

/// The part of the CloudWatch sink that talks to the service.
///
/// Split out so a publish can be spawned: a task needs to own what it uses, and
/// cloning one [`Arc`] costs an increment rather than an allocation per request.
#[derive(Debug)]
struct CloudWatchWriter {
    client: Arc<aws_sdk_cloudwatch::Client>,
    namespace: String,
}

impl CloudWatchWriter {
    /// Sends `datums`, splitting a batch too large for one call.
    async fn send(&self, datums: Vec<aws_sdk_cloudwatch::types::MetricDatum>) {
        for call in calls(datums) {
            self.put(call).await;
        }
    }

    /// Sends one call's worth, reporting rather than discarding a failure.
    ///
    /// A silently dropped metric looks exactly like a healthy service, which is the
    /// worst way for monitoring to fail.
    async fn put(&self, datums: Vec<aws_sdk_cloudwatch::types::MetricDatum>) {
        if datums.is_empty() {
            return;
        }
        let count = datums.len();
        if let Err(e) = self
            .client
            .put_metric_data()
            .namespace(&self.namespace)
            .set_metric_data(Some(datums))
            .send()
            .await
        {
            eprintln!(
                "metrics: could not send {count} data points to CloudWatch namespace {}: {}",
                self.namespace,
                crate::io::describe_aws_error(&e)
            );
        }
    }
}

/// Divides a batch into calls `PutMetricData` will accept.
///
/// Belt and braces: the accumulator's capacity is already rounded so that a batch
/// lands on the limit, and this keeps an over-long batch from being lost entirely if
/// that arithmetic is ever changed. The API rejects an over-long call **whole**, so
/// getting it wrong costs every data point in the batch rather than the excess.
///
/// A free function so the division is testable without a network or a live account,
/// which is the only part of sending that has a decision in it. The common case is one
/// call carrying the original buffer, so nothing is copied.
fn calls(
    datums: Vec<aws_sdk_cloudwatch::types::MetricDatum>,
) -> Vec<Vec<aws_sdk_cloudwatch::types::MetricDatum>> {
    if datums.is_empty() {
        return Vec::new();
    }
    if datums.len() <= CLOUDWATCH_MAX_DATUMS {
        return vec![datums];
    }
    datums
        .chunks(CLOUDWATCH_MAX_DATUMS)
        .map(<[aws_sdk_cloudwatch::types::MetricDatum]>::to_vec)
        .collect()
}

/// Amazon CloudWatch.
///
/// Buffers data points and sends them in batches, because `PutMetricData` is charged
/// per call and rate limited. One buffer serves every request handler, so the send
/// interval is a property of the service's throughput rather than of how many
/// handlers happen to be running.
#[derive(Debug)]
pub(crate) struct CloudWatchMetrics {
    writer: Arc<CloudWatchWriter>,
    accumulator: Accumulator<aws_sdk_cloudwatch::types::MetricDatum>,
}

impl CloudWatchMetrics {
    /// `batches` is scored batches, not data points.
    ///
    /// The unit is batches so that one `metrics.batch_size` means the same thing
    /// here as it does on stderr; an earlier version counted data points here and
    /// batches there, so the same number meant 500 summaries in one place and 167 in
    /// the other. Each batch contributes one data point per stage, and
    /// `PutMetricData` takes at most 1000 of those, so the effective batch count is
    /// capped at 333 however large a number is configured — which is what
    /// [`Self::batch_size`] reports.
    pub(crate) fn new(
        client: Arc<aws_sdk_cloudwatch::Client>,
        namespace: impl Into<String>,
        batches: usize,
        max_in_flight: usize,
    ) -> Self {
        let datums = batches
            .saturating_mul(STAGES.len())
            .clamp(STAGES.len(), CLOUDWATCH_MAX_BATCHED_DATUMS);
        Self {
            writer: Arc::new(CloudWatchWriter {
                client,
                namespace: namespace.into(),
            }),
            accumulator: Accumulator::new("metrics", datums, max_in_flight),
        }
    }
}

#[tonic::async_trait]
impl MetricsSink for CloudWatchMetrics {
    fn backend(&self) -> &str {
        &self.writer.namespace
    }

    fn batch_size(&self) -> usize {
        self.accumulator.capacity() / STAGES.len()
    }

    fn record(&self, model_id: &str, timings: &InferenceMicros) {
        let values = [
            timings.vec_time,
            timings.tensor_time,
            timings.inference_time,
        ];
        // A dimension rather than a metric name per model, so one alarm and one graph
        // cover every arm, and a roll-out does not need new dashboards to be watched.
        let model = aws_sdk_cloudwatch::types::Dimension::builder()
            .name("ModelId")
            .value(model_id)
            .build();
        let datums = STAGES.iter().zip(values).map(|(name, micros)| {
            aws_sdk_cloudwatch::types::MetricDatum::builder()
                .metric_name(*name)
                .dimensions(model.clone())
                .value(micros as f64)
                .unit(aws_sdk_cloudwatch::types::StandardUnit::Microseconds)
                .build()
        });

        let writer = Arc::clone(&self.writer);
        self.accumulator
            .add(datums, move |batch| async move { writer.send(batch).await });
    }

    async fn flush(&self) {
        let writer = Arc::clone(&self.writer);
        self.accumulator
            .flush(move |batch| async move { writer.send(batch).await })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timings(vec: u128, tensor: u128, inference: u128) -> InferenceMicros {
        InferenceMicros {
            vec_time: vec,
            tensor_time: tensor,
            inference_time: inference,
        }
    }

    /// The model a single-model test records under. Any name will do; what matters is
    /// that it is the same one the assertion looks up.
    const MODEL: &str = "test-model";

    /// Batches held for `model`, or zero when nothing has been recorded for it.
    fn batches(sink: &StderrMetrics, model: &str) -> u64 {
        sink.lock().get(model).map_or(0, |state| state.batches)
    }

    #[tokio::test]
    async fn the_stderr_sink_aggregates_rather_than_logging_every_batch() {
        let sink = StderrMetrics::new(3);
        sink.record(MODEL, &timings(1, 2, 3));
        sink.record(MODEL, &timings(1, 2, 3));
        assert_eq!(batches(&sink, MODEL), 2, "still holding at two of three");

        sink.record(MODEL, &timings(1, 2, 3));
        assert_eq!(
            batches(&sink, MODEL),
            0,
            "the interval should reset after emitting, so nothing is double-counted"
        );
    }

    /// The reason the dimension exists: a roll-out's arms must not average together,
    /// or the difference the roll-out is measuring disappears into one number.
    #[tokio::test]
    async fn two_models_are_aggregated_separately() {
        let sink = StderrMetrics::new(1000);
        sink.record("control", &timings(0, 0, 10));
        sink.record("control", &timings(0, 0, 30));
        sink.record("candidate", &timings(0, 0, 100));

        assert_eq!(batches(&sink, "control"), 2);
        assert_eq!(batches(&sink, "candidate"), 1);
        let state = sink.lock();
        assert_eq!(
            state["control"].totals[2], 40,
            "control keeps its own total"
        );
        assert_eq!(
            state["candidate"].totals[2], 100,
            "the candidate's slower batch must not be charged to the control"
        );
        assert_eq!(state["control"].max_inference, 30);
        assert_eq!(state["candidate"].max_inference, 100);
    }

    /// Each arm's line covers the same number of batches, so a 10% candidate emits a
    /// tenth as often rather than emitting an average over a handful.
    #[tokio::test]
    async fn the_interval_is_counted_per_model() {
        let sink = StderrMetrics::new(2);
        sink.record("control", &timings(0, 0, 1));
        sink.record("candidate", &timings(0, 0, 1));
        assert_eq!(
            (batches(&sink, "control"), batches(&sink, "candidate")),
            (1, 1),
            "one batch each is not two batches for one model"
        );

        sink.record("control", &timings(0, 0, 1));
        assert_eq!(
            batches(&sink, "control"),
            0,
            "control reached two and emitted"
        );
        assert_eq!(
            batches(&sink, "candidate"),
            1,
            "the candidate still holds one"
        );
    }
    /// A mean hides the tail, and the tail is where a latency investigation starts.
    #[tokio::test]
    async fn the_maximum_is_kept_not_averaged_away() {
        let sink = StderrMetrics::new(1000);
        sink.record(MODEL, &timings(0, 0, 5));
        sink.record(MODEL, &timings(0, 0, 900));
        sink.record(MODEL, &timings(0, 0, 7));

        let state = sink.lock();
        assert_eq!(state[MODEL].max_inference, 900);
        assert_eq!(state[MODEL].totals[2], 912);
    }
    /// Whatever is buffered at shutdown is exactly the window where the metrics matter
    /// most, so it must not be dropped.
    #[tokio::test]
    async fn flushing_emits_a_partial_interval() {
        let sink = StderrMetrics::new(1000);
        sink.record(MODEL, &timings(1, 1, 1));
        sink.flush().await;
        assert_eq!(batches(&sink, MODEL), 0);
    }

    /// Both arms' partial intervals, not just whichever was seen last.
    #[tokio::test]
    async fn flushing_emits_every_model() {
        let sink = StderrMetrics::new(1000);
        sink.record("control", &timings(1, 1, 1));
        sink.record("candidate", &timings(1, 1, 1));
        sink.flush().await;
        assert_eq!(batches(&sink, "control"), 0);
        assert_eq!(batches(&sink, "candidate"), 0);
    }
    /// The service can stop before any batch arrives.
    #[tokio::test]
    async fn flushing_nothing_is_harmless() {
        StderrMetrics::new(10).flush().await;
        assert_eq!(batches(&StderrMetrics::new(10), MODEL), 0);
    }

    #[tokio::test]
    async fn a_zero_interval_does_not_divide_by_zero() {
        let sink = StderrMetrics::new(0);
        sink.record(MODEL, &timings(1, 1, 1));
        assert_eq!(batches(&sink, MODEL), 0);
        assert_eq!(sink.batch_size(), 1);
    }
    /// The property the request path relies on: one sink, every handler, no `&mut`.
    #[tokio::test]
    async fn a_sink_is_usable_from_several_handlers_at_once() {
        let sink: Arc<dyn MetricsSink> = Arc::new(StderrMetrics::new(100));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let sink = Arc::clone(&sink);
            handles.push(tokio::spawn(async move {
                sink.record(MODEL, &timings(1, 1, 1));
            }));
        }
        for handle in handles {
            handle.await.expect("handler");
        }
        assert_eq!(sink.backend(), "stderr");
    }
    /// The names end up in a dashboard, so a duplicate would silently merge two
    /// different measurements into one series.
    #[test]
    fn every_stage_has_a_name_and_no_two_share_one() {
        assert_eq!(STAGES.len(), 3);
        let mut sorted = STAGES;
        sorted.sort_unstable();
        let mut deduped = sorted.to_vec();
        deduped.dedup();
        assert_eq!(
            deduped.len(),
            STAGES.len(),
            "two stages share a metric name"
        );
    }

    /// A CloudWatch sink with no usable credentials.
    ///
    /// Enough to test buffering, which is the part with logic in it. Sending is not
    /// exercised: that would need either a live account or an HTTP fake, and
    /// neither belongs in a unit test.
    fn offline_cloudwatch(batches: usize) -> CloudWatchMetrics {
        let config = aws_sdk_cloudwatch::Config::builder()
            .behavior_version_latest()
            .region(aws_sdk_cloudwatch::config::Region::new("us-east-1"))
            .credentials_provider(aws_sdk_cloudwatch::config::Credentials::for_tests())
            .build();
        CloudWatchMetrics::new(
            Arc::new(aws_sdk_cloudwatch::Client::from_conf(config)),
            "HusharTest",
            batches,
            2,
        )
    }
    /// Each batch contributes one data point per stage, so a size of two batches is six
    /// data points. `PutMetricData` is charged per call, which is why this batches at
    /// all. Reaching the size takes the buffer for sending, leaving it empty.
    #[tokio::test]
    async fn cloudwatch_buffers_until_it_reaches_the_configured_batch_count() {
        let sink = offline_cloudwatch(2);
        sink.record(MODEL, &timings(1, 2, 3));
        assert_eq!(
            sink.accumulator.buffered(),
            3,
            "one batch, three data points"
        );

        sink.record(MODEL, &timings(1, 2, 3));
        assert_eq!(
            sink.accumulator.buffered(),
            0,
            "hitting the batch count should drain the buffer"
        );
    }
    /// The unit is batches on both sinks, which is the whole reason a single
    /// `metrics.batch_size` can feed either without meaning two different things.
    #[tokio::test]
    async fn the_configured_size_is_batches_on_both_sinks() {
        assert_eq!(offline_cloudwatch(100).batch_size(), 100);
        assert_eq!(StderrMetrics::new(100).batch_size(), 100);
    }
    /// CloudWatch rejects a call carrying more than 1000 data points, and rejects it
    /// whole -- so the batch count is capped at what fits, and reports the cap rather
    /// than the number it was given.
    #[tokio::test]
    async fn a_batch_count_beyond_what_cloudwatch_accepts_is_capped() {
        assert_eq!(
            offline_cloudwatch(50_000).batch_size(),
            CLOUDWATCH_MAX_BATCHED_DATUMS / STAGES.len(),
            "1000 data points is 333 batches, whatever was asked for"
        );
        assert_eq!(offline_cloudwatch(0).batch_size(), 1, "zero sends eagerly");
    }

    /// **The regression this exists for.** The accumulator sends on "at least
    /// capacity", and a batch adds three data points at a time, so a capacity that is
    /// not a multiple of three is first reached one batch late — 1000 is reached at
    /// 1002, which is over the limit and so is rejected *whole*. At the default batch
    /// size that silently published nothing at all, leaving only a recurring error.
    ///
    /// Asserted on the arithmetic and on the boundary, because the failure was
    /// invisible from outside: buffering, batching and flushing all behaved.
    #[tokio::test]
    async fn a_publish_can_never_exceed_what_the_api_accepts() {
        for batches in [1, 2, 100, 333, 334, 500, 50_000] {
            let sink = offline_cloudwatch(batches);
            let capacity = sink.accumulator.capacity();

            assert!(
                capacity.is_multiple_of(STAGES.len()),
                "a capacity of {capacity} is not a whole number of batches, so the \
                 buffer would overshoot it"
            );

            // Fill to exactly the reported batch count and confirm it fired there.
            for _ in 0..sink.batch_size() {
                sink.record(MODEL, &timings(1, 1, 1));
            }
            assert_eq!(
                sink.accumulator.buffered(),
                0,
                "{batches} batches configured: the send should trigger at the {} \
                 batches reported, not later",
                sink.batch_size()
            );
            assert!(
                capacity <= CLOUDWATCH_MAX_DATUMS,
                "a send of {capacity} data points would be rejected whole"
            );
        }
    }
    /// And if that arithmetic is ever changed, the writer still cannot build a call the
    /// service refuses: an over-long batch is sliced rather than lost.
    ///
    /// Exercises `calls` itself rather than reimplementing the slicing in the test,
    /// which would assert on the test's own arithmetic and pass whatever the code did.
    #[test]
    fn an_over_long_batch_is_split_into_calls_the_api_accepts() {
        let datum = |i: usize| {
            aws_sdk_cloudwatch::types::MetricDatum::builder()
                .metric_name(STAGES[i % STAGES.len()])
                .value(i as f64)
                .build()
        };

        for total in [0, 1, 999, 1000, 1001, 2001] {
            let calls = calls((0..total).map(datum).collect());
            assert!(
                calls.iter().all(|c| c.len() <= CLOUDWATCH_MAX_DATUMS),
                "{total} data points produced a call over the limit: {:?}",
                calls.iter().map(Vec::len).collect::<Vec<_>>()
            );
            assert_eq!(
                calls.iter().map(Vec::len).sum::<usize>(),
                total,
                "{total} data points: splitting must not drop or duplicate any"
            );
            assert!(
                !calls.iter().any(Vec::is_empty),
                "{total} data points produced an empty call, which is a wasted request"
            );
        }

        assert_eq!(calls(Vec::new()).len(), 0, "nothing to send is no calls");
        assert_eq!(
            calls((0..CLOUDWATCH_MAX_DATUMS).map(datum).collect()).len(),
            1,
            "a batch at the limit is one call, not two"
        );
        assert_eq!(
            calls((0..CLOUDWATCH_MAX_DATUMS + 1).map(datum).collect()).len(),
            2,
            "one over the limit is two"
        );
    }

    #[tokio::test]
    async fn the_namespace_is_what_the_banner_reports() {
        assert_eq!(offline_cloudwatch(10).backend(), "HusharTest");
    }
    /// Recording must not wait on CloudWatch; with credentials that cannot work, an
    /// awaited publish would spend its retries inside this loop.
    #[tokio::test]
    async fn recording_returns_without_waiting_for_the_publish() {
        let sink = offline_cloudwatch(1);
        let start = std::time::Instant::now();
        for _ in 0..50 {
            sink.record(MODEL, &timings(1, 1, 1));
        }
        let recording = start.elapsed();
        assert!(
            recording < std::time::Duration::from_millis(50),
            "50 records took {recording:?}, so recording is waiting on the publish"
        );
    }
}
