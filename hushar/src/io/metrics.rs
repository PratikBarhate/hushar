// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Where inference timings go.
//!
//! Today that means stderr or Amazon CloudWatch. As with [`BlobStore`](super::BlobStore),
//! adding a backend is one implementation of the trait, and nothing above this module
//! changes.
//!
//! Recording happens on the request path and never blocks: a lock and a few additions
//! for stderr, a lock and a push for CloudWatch, with the publish spawned once a batch
//! is full. See [`accumulator`](super::accumulator) for how memory stays bounded.
//!
//! Timings are never sampled. They are already an aggregate of three microsecond
//! figures per batch, whatever the batch's size.

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
    /// buffer it spawns the publish. Nothing here awaits a network call.
    ///
    /// # Arguments
    ///
    /// * `model_id` - the model that produced the timings; timings are aggregated per
    ///   model, so two models never average together
    /// * `timings` - one batch's per-stage durations, in microseconds
    ///
    /// # Panics
    ///
    /// If called outside a tokio runtime, since the publish is spawned.
    fn record(&self, model_id: &str, timings: &InferenceMicros);

    /// Pushes anything buffered and waits for the sends already in the air.
    ///
    /// Called once at shutdown, so a batching sink does not lose what it was holding.
    async fn flush(&self);
}

/// Metric names for the three timings a batch produces, shared by every sink.
const STAGES: [&str; 3] = ["VectorizationTime", "ToTensorTime", "InferenceTime"];

/// Data points `PutMetricData` accepts in one call.
///
/// Exceeding it rejects the call whole, losing every data point in it.
const CLOUDWATCH_MAX_DATUMS: usize = 1000;

/// The largest capacity that both fits one call and lands on a batch boundary.
///
/// The accumulator sends on "at least capacity" rather than exactly capacity, and a
/// batch contributes one data point per stage, so a capacity that is not a whole number
/// of batches is first reached past the limit — and such a call is rejected whole.
/// Rounding down keeps the buffer on a batch boundary, which also makes
/// [`CloudWatchMetrics::batch_size`] the number of batches that really triggers a send.
const CLOUDWATCH_MAX_BATCHED_DATUMS: usize =
    CLOUDWATCH_MAX_DATUMS - CLOUDWATCH_MAX_DATUMS % STAGES.len();

/// Aggregates timings and writes periodic summaries to stderr.
///
/// The default, since it needs no credentials and no network. Running totals are a fixed
/// size whatever the batch size, so this is the one sink with no accumulator: there is no
/// buffer to bound and nothing to spawn.
#[derive(Debug)]
pub(crate) struct StderrMetrics {
    every: usize,
    /// One aggregate per model, so two models are summarised separately.
    state: std::sync::Mutex<std::collections::BTreeMap<String, Aggregate>>,
}

/// Running totals between summaries: a batch count, and microseconds per stage.
#[derive(Debug, Default)]
struct Aggregate {
    batches: u64,
    totals: [u128; 3],
    max_inference: u128,
}

impl StderrMetrics {
    /// Summarises every `every` batches, counted per model. Zero is treated as one.
    pub(crate) fn new(every: usize) -> Self {
        Self {
            every: every.max(1),
            state: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    /// Recovers the guard even if a previous holder panicked.
    fn lock(&self) -> std::sync::MutexGuard<'_, std::collections::BTreeMap<String, Aggregate>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes one model's running totals to stderr, naming the model, and clears them.
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
/// Held behind an [`Arc`] so a spawned publish can own what it sends.
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

    /// Sends one call's worth, reporting a failure on stderr rather than discarding it.
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
/// Each call carries at most `CLOUDWATCH_MAX_DATUMS` data points, no call is empty, and
/// no data point is dropped or duplicated. An empty batch produces no calls, and a batch
/// at or under the limit is returned as one call without copying.
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
/// Buffers data points and sends them in batches, since `PutMetricData` is charged per
/// call and rate limited. One buffer serves every request handler. The model is carried
/// as a `ModelId` dimension rather than in the metric name, so one alarm and one graph
/// cover every model.
#[derive(Debug)]
pub(crate) struct CloudWatchMetrics {
    writer: Arc<CloudWatchWriter>,
    accumulator: Accumulator<aws_sdk_cloudwatch::types::MetricDatum>,
}

impl CloudWatchMetrics {
    /// Buffers `batches` scored batches — not data points — before each send.
    ///
    /// Each batch contributes one data point per stage, and `PutMetricData` takes at
    /// most 1000 of them, so the effective batch count is capped at 333 however large a
    /// number is configured; [`Self::batch_size`] reports the cap. Zero sends eagerly,
    /// on every batch.
    ///
    /// # Arguments
    ///
    /// * `client` - the CloudWatch client
    /// * `namespace` - the namespace data points are published under
    /// * `batches` - scored batches buffered before one send
    /// * `max_in_flight` - publishes allowed to be in the air at once
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

    /// The model a single-model test records under.
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

    /// Two models keep separate totals and maxima.
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

    /// The summary interval is counted per model, not across all of them.
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
    /// The maximum inference time is kept alongside the running totals.
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
    /// Flushing emits an interval that has not filled yet.
    #[tokio::test]
    async fn flushing_emits_a_partial_interval() {
        let sink = StderrMetrics::new(1000);
        sink.record(MODEL, &timings(1, 1, 1));
        sink.flush().await;
        assert_eq!(batches(&sink, MODEL), 0);
    }

    /// Flushing emits every model's partial interval, not just the last one seen.
    #[tokio::test]
    async fn flushing_emits_every_model() {
        let sink = StderrMetrics::new(1000);
        sink.record("control", &timings(1, 1, 1));
        sink.record("candidate", &timings(1, 1, 1));
        sink.flush().await;
        assert_eq!(batches(&sink, "control"), 0);
        assert_eq!(batches(&sink, "candidate"), 0);
    }
    /// Flushing before any batch has been recorded is harmless.
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
    /// One sink records from several handlers at once, with no `&mut`.
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
    /// Every stage has a metric name and no two stages share one.
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

    /// A CloudWatch sink with test credentials, for exercising buffering without sending.
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
    /// Buffering holds three data points per batch and drains at the configured count.
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
    /// Both sinks report their configured size in batches.
    #[tokio::test]
    async fn the_configured_size_is_batches_on_both_sinks() {
        assert_eq!(offline_cloudwatch(100).batch_size(), 100);
        assert_eq!(StderrMetrics::new(100).batch_size(), 100);
    }
    /// A batch count beyond what one call accepts is capped, and the cap is reported.
    #[tokio::test]
    async fn a_batch_count_beyond_what_cloudwatch_accepts_is_capped() {
        assert_eq!(
            offline_cloudwatch(50_000).batch_size(),
            CLOUDWATCH_MAX_BATCHED_DATUMS / STAGES.len(),
            "1000 data points is 333 batches, whatever was asked for"
        );
        assert_eq!(offline_cloudwatch(0).batch_size(), 1, "zero sends eagerly");
    }

    /// Capacity is always a whole number of batches, so a publish never exceeds the
    /// limit and the send triggers at the reported batch count.
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
    /// An over-long batch is split into calls the API accepts, dropping none.
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
    /// Recording returns without waiting for the publish it spawns.
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
