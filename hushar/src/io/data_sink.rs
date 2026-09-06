// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Where the data a request produces goes.
//!
//! Today that is the inference log: the features as received and the scores they
//! produced, as batched protobuf objects in a blob store or as records on a Kinesis
//! data stream. The trait is [`DataSink`] rather than `LogSink` because the inference
//! log is not the only data a serving host has worth shipping — application logs to
//! CloudWatch Logs are the obvious next one — and a name that says "log" would either
//! have to be reused for something it does not describe or be renamed later.
//!
//! A second kind of payload needs its own trait or a generic one; the name is what is
//! being reserved here, not the shape.
//!
//! # No sidecar
//!
//! A sink used to be drained by workers on their own runtime, fed through a bounded
//! channel. Now the request path calls [`DataSink::record`] directly, which appends to
//! an [`Accumulator`](super::accumulator::Accumulator) and returns. Nothing awaits a
//! network call on a request, and no thread is reserved away from serving. See that
//! module for what each request pays and how memory stays bounded.
//!
//! # Adding a destination
//!
//! 1. Implement [`DataSink`].
//! 2. Add a scheme to [`DataDestination::parse`].
//! 3. Add an arm to `build_data_sink` in `main`.

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use aws_sdk_kinesis::types::{PutRecordsRequestEntry, PutRecordsResultEntry};
use chrono::{Datelike, Timelike};
use hushar::hushar_proto::{InferenceLogBatch, InferenceLogs};
use prost::Message;

use crate::io::accumulator::Accumulator;
use crate::io::{BlobStore, FileReaderResult, Location, describe_aws_error};

/// Where inference logs are sent.
///
/// Variants:
/// - `Blob` — batched protobuf objects beneath a prefix.
/// - `Kinesis` — records on a Kinesis data stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DataDestination {
    Blob(Location),
    Kinesis { stream: String },
}

impl DataDestination {
    /// Parses a destination URI.
    ///
    /// `kinesis://stream-name` selects the stream; anything else is a blob-store
    /// location and follows [`Location::parse`], so `s3://bucket/prefix` and bare
    /// filesystem paths both keep working.
    pub(crate) fn parse(uri: &str) -> FileReaderResult<Self> {
        if let Some(stream) = uri.strip_prefix("kinesis://") {
            let stream = stream.trim_matches('/');
            if stream.is_empty() {
                return Err(format!("{uri:?} has no stream name").into());
            }
            if stream.contains('/') {
                return Err(format!(
                    "{uri:?} looks like a path, but a Kinesis destination is just a \
                     stream name, as in kinesis://my-stream"
                )
                .into());
            }
            return Ok(Self::Kinesis {
                stream: stream.to_owned(),
            });
        }
        Ok(Self::Blob(Location::parse(uri)?))
    }

    /// The destination as a URI, for banners and errors.
    pub(crate) fn uri(&self) -> String {
        match self {
            Self::Blob(location) => location.uri(),
            Self::Kinesis { stream } => format!("kinesis://{stream}"),
        }
    }
}

/// How many requests reach the data sink.
///
/// The cheapest lever there is over the memory the inference log costs, and the only
/// one that also removes work: an unsampled request never clones its feature maps,
/// because the decision is made before scoring rather than after. See
/// [`FeatureLogging`](crate::inference::scoring::FeatureLogging).
///
/// **Sampling is per request, not per row.** A request that is kept is kept whole, so
/// a consumer joining rows back to one `request_id` never sees a partial batch — which
/// is what makes the sample usable for debugging one prediction rather than only for
/// aggregates.
///
/// Counting rather than randomising, for three reasons: it needs no random-number
/// generator on the request path, it keeps exactly the configured fraction instead of
/// approaching it, and it cannot degenerate on request ids that are not uniform.
#[derive(Debug)]
pub(crate) struct Sampler {
    /// Keep one request in this many. Zero keeps none.
    every: u64,
    seen: AtomicU64,
}

impl Sampler {
    /// `rate` is the fraction of requests to record, in `0.0..=1.0`.
    ///
    /// Realised as one request in `n`, so a rate that is not a unit fraction becomes
    /// the nearest one: 0.15 is kept as one in 7, which is 14.3%. [`Self::describe`]
    /// reports what was chosen rather than what was asked for, because a sampling rate
    /// that quietly differs from the configured one is discovered while wondering
    /// where the logs went.
    pub(crate) fn new(rate: f64) -> Self {
        // NaN is checked first: it compares false against everything, so it would
        // otherwise fall through to the division and produce a nonsense interval.
        let every = if rate.is_nan() || rate <= 0.0 {
            0
        } else if rate >= 1.0 {
            1
        } else {
            (1.0 / rate).round() as u64
        };
        Self {
            every,
            seen: AtomicU64::new(0),
        }
    }

    /// Whether this request's features are recorded.
    ///
    /// One relaxed increment in the common sampled case, and not even that when
    /// everything or nothing is kept.
    pub(crate) fn sample(&self) -> bool {
        match self.every {
            0 => false,
            1 => true,
            n => self.seen.fetch_add(1, Ordering::Relaxed).is_multiple_of(n),
        }
    }

    /// What was actually chosen, for the startup banner.
    pub(crate) fn describe(&self) -> String {
        match self.every {
            0 => "no requests".to_owned(),
            1 => "every request".to_owned(),
            n => format!("1 request in {n} ({:.1}%)", 100.0 / n as f64),
        }
    }
}

/// Somewhere to send the data a request produced.
///
/// Implementations are shared across request handlers behind an [`Arc`], so every
/// method takes `&self`. Batching belongs here rather than above, so that each
/// destination batches the way its own API wants.
#[tonic::async_trait]
pub(crate) trait DataSink: Debug + Send + Sync {
    /// Which destination this is, for the startup banner.
    fn backend(&self) -> &str;

    /// Batches accumulated before one send, after any clamping the API forces.
    ///
    /// Reported at startup so a value the destination's own limit reduced is visible
    /// rather than surprising.
    fn batch_size(&self) -> usize;

    /// Accepts one batch, on the request path.
    ///
    /// Synchronous and non-blocking on purpose: it appends to a buffer, and on the
    /// request that fills it spawns the send. A sink whose `record` awaited a network
    /// call would put that latency on one request in every `batch_size`.
    ///
    /// # Panics
    ///
    /// If called outside a tokio runtime, since the send is spawned.
    fn record(&self, batch: InferenceLogBatch);

    /// Sends anything buffered and waits for the sends already in the air.
    ///
    /// Called once at shutdown. Without it a partial buffer and every detached send
    /// would go with the runtime, which is when a log is most likely to be the only
    /// record of what the service was doing.
    async fn flush(&self);
}

/// The part of a blob sink that talks to the store.
///
/// Split out so a send can be spawned: a task needs to own what it uses, and cloning
/// one [`Arc`] costs an increment rather than the two allocations that cloning a store
/// handle and a prefix would.
#[derive(Debug)]
struct BlobWriter {
    store: Arc<dyn BlobStore>,
    prefix: Location,
}

impl BlobWriter {
    /// Encodes and writes one object, reporting a failure rather than dropping it
    /// silently.
    ///
    /// A write that fails quietly leaves a service that looks healthy and logs
    /// nothing, which is discovered weeks later by whoever needed the data.
    async fn write(&self, batches: Vec<InferenceLogBatch>) {
        if batches.is_empty() {
            return;
        }
        let rows: usize = batches.iter().map(|b| b.inference_log_rows.len()).sum();
        let logs = InferenceLogs {
            inference_log_batches: batches,
        };
        let target = self.prefix.join(&time_partitioned_key());
        if let Err(e) = self.store.put(&target, logs.encode_to_vec()).await {
            eprintln!(
                "logs: lost {rows} rows, could not write {}: {e}",
                target.uri()
            );
        }
    }
}

/// Batched protobuf objects in a blob store.
///
/// The default. Objects are named by time so they can be queried by partition without
/// listing everything, and batched because one object per request would be both slow
/// and expensive on any object store.
#[derive(Debug)]
pub(crate) struct BlobDataSink {
    writer: Arc<BlobWriter>,
    accumulator: Accumulator<InferenceLogBatch>,
}

impl BlobDataSink {
    /// `capacity` is batches per object, and `max_in_flight` writes at once.
    ///
    /// Nothing clamps the capacity here: an object store takes an object of whatever
    /// size, so this is the one sink where the batch size is purely a memory decision.
    pub(crate) fn new(
        store: Arc<dyn BlobStore>,
        prefix: Location,
        capacity: usize,
        max_in_flight: usize,
    ) -> Self {
        Self {
            writer: Arc::new(BlobWriter { store, prefix }),
            accumulator: Accumulator::new("logs", capacity, max_in_flight),
        }
    }
}

#[tonic::async_trait]
impl DataSink for BlobDataSink {
    /// Deferred to the store, so "Amazon S3" and "local filesystem" read the same here
    /// as they do for anything else reading objects.
    fn backend(&self) -> &str {
        self.writer.store.backend()
    }

    fn batch_size(&self) -> usize {
        self.accumulator.capacity()
    }

    fn record(&self, batch: InferenceLogBatch) {
        let writer = Arc::clone(&self.writer);
        self.accumulator.add([batch], move |batches| async move {
            writer.write(batches).await
        });
    }

    async fn flush(&self) {
        let writer = Arc::clone(&self.writer);
        self.accumulator
            .flush(move |batches| async move { writer.write(batches).await })
            .await;
    }
}

/// A time-partitioned object name, unique per write.
///
/// Hive-style partitioning so the logs can be queried by time without listing every
/// object, and a UUID so concurrent writes never collide on a name.
fn time_partitioned_key() -> String {
    let now = chrono::Utc::now();
    format!(
        "year={}/month={:02}/day={:02}/hour={:02}/mi={:02}/{}_{}.pb",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute(),
        uuid::Uuid::new_v4(),
        now.timestamp_micros()
    )
}

/// Records `PutRecords` accepts in one call.
const KINESIS_MAX_RECORDS: usize = 500;

/// Bytes one record may carry, partition key included.
const KINESIS_MAX_RECORD_BYTES: usize = 10 * 1024 * 1024;

/// Attempts made before a batch is reported lost.
///
/// Bounded on purpose. Retrying indefinitely would hold a send slot until the
/// accumulator sheds a batch, turning a throttled stream into a wider outage of
/// observability; better to report the loss and free the slot.
const KINESIS_MAX_ATTEMPTS: usize = 3;

/// The part of a Kinesis sink that talks to the stream.
#[derive(Debug)]
struct KinesisWriter {
    client: Arc<aws_sdk_kinesis::Client>,
    stream: String,
}

impl KinesisWriter {
    /// Sends `batches` as records, retrying the ones the stream rejected.
    ///
    /// The part that is easy to get wrong: `PutRecords` answers **200 even when
    /// individual records were rejected**, reporting them positionally in the response
    /// body. A producer that checks only the `Result` loses rows while reporting
    /// success, so the body is inspected and only the rejected records are retried.
    ///
    /// A whole-request failure leaves every record pending.
    async fn send(&self, batches: Vec<InferenceLogBatch>) {
        let mut pending: Vec<PutRecordsRequestEntry> = Vec::with_capacity(batches.len());
        for batch in batches {
            match to_entry(&batch) {
                Ok(entry) => pending.push(entry),
                Err(reason) => eprintln!(
                    "logs: dropped request {:?} with {} rows: {reason}",
                    batch.request_id,
                    batch.inference_log_rows.len()
                ),
            }
        }

        for attempt in 1..=KINESIS_MAX_ATTEMPTS {
            if pending.is_empty() {
                return;
            }

            let response = match self
                .client
                .put_records()
                .stream_name(&self.stream)
                .set_records(Some(pending.clone()))
                .send()
                .await
            {
                Ok(response) => response,
                Err(e) => {
                    eprintln!(
                        "logs: attempt {attempt}/{KINESIS_MAX_ATTEMPTS} to stream {} failed: {}",
                        self.stream,
                        describe_aws_error(&e)
                    );
                    // Not after the last attempt: the loop is about to end, so the
                    // wait would only hold a send slot while achieving nothing.
                    if attempt < KINESIS_MAX_ATTEMPTS {
                        backoff(attempt).await;
                    }
                    continue;
                }
            };

            let (retry, reasons) = failed_entries(pending, response.records());
            if retry.is_empty() {
                return;
            }
            eprintln!(
                "logs: {} of {} records rejected by stream {} on attempt \
                 {attempt}/{KINESIS_MAX_ATTEMPTS}: {}",
                retry.len(),
                retry.len() + response.records().len().saturating_sub(retry.len()),
                self.stream,
                summarise(&reasons)
            );
            pending = retry;
            if attempt < KINESIS_MAX_ATTEMPTS {
                backoff(attempt).await;
            }
        }

        if !pending.is_empty() {
            eprintln!(
                "logs: giving up on {} records after {KINESIS_MAX_ATTEMPTS} attempts \
                 to stream {}; those rows are lost",
                pending.len(),
                self.stream
            );
        }
    }
}

/// A Kinesis data stream.
#[derive(Debug)]
pub(crate) struct KinesisDataSink {
    writer: Arc<KinesisWriter>,
    accumulator: Accumulator<InferenceLogBatch>,
}

impl KinesisDataSink {
    /// `capacity` is batches, clamped to the 500 records `PutRecords` accepts, because
    /// one batch becomes one record. A larger buffer would build a request the service
    /// rejects whole, losing everything in it rather than some of it.
    pub(crate) fn new(
        client: Arc<aws_sdk_kinesis::Client>,
        stream: impl Into<String>,
        capacity: usize,
        max_in_flight: usize,
    ) -> Self {
        Self {
            writer: Arc::new(KinesisWriter {
                client,
                stream: stream.into(),
            }),
            accumulator: Accumulator::new(
                "logs",
                capacity.clamp(1, KINESIS_MAX_RECORDS),
                max_in_flight,
            ),
        }
    }
}

#[tonic::async_trait]
impl DataSink for KinesisDataSink {
    fn backend(&self) -> &str {
        "Amazon Kinesis"
    }

    fn batch_size(&self) -> usize {
        self.accumulator.capacity()
    }

    fn record(&self, batch: InferenceLogBatch) {
        let writer = Arc::clone(&self.writer);
        self.accumulator.add(
            [batch],
            move |batches| async move { writer.send(batches).await },
        );
    }

    async fn flush(&self) {
        let writer = Arc::clone(&self.writer);
        self.accumulator
            .flush(move |batches| async move { writer.send(batches).await })
            .await;
    }
}

/// Turns one batch into one stream record.
///
/// The partition key decides the shard, and Kinesis hashes it, so `request_id` gives
/// an even spread while keeping one request's rows together for a consumer. Using
/// the model id instead would put every record on one shard, which is the classic
/// way to throttle a stream that is nominally wide enough.
///
/// The limit covers the key as well as the payload.
fn to_entry(batch: &InferenceLogBatch) -> Result<PutRecordsRequestEntry, String> {
    let data = batch.encode_to_vec();
    let key = partition_key(&batch.request_id);

    let size = data.len() + key.len();
    if size > KINESIS_MAX_RECORD_BYTES {
        return Err(format!(
            "encoded to {size} bytes, over the {KINESIS_MAX_RECORD_BYTES}-byte \
             per-record limit; reduce the batch size"
        ));
    }

    PutRecordsRequestEntry::builder()
        .data(aws_sdk_kinesis::primitives::Blob::new(data))
        .partition_key(key)
        .build()
        .map_err(|e| format!("cannot build a stream record: {e}"))
}

/// A partition key that Kinesis will accept.
///
/// The service requires between 1 and 256 characters, so an absent request id and an
/// over-long one both have to be handled rather than passed through — either would
/// be rejected for the whole request, not just that record.
fn partition_key(request_id: &str) -> String {
    const MAX: usize = 256;
    let trimmed = request_id.trim();
    if trimmed.is_empty() {
        return uuid::Uuid::new_v4().to_string();
    }
    if trimmed.len() <= MAX {
        return trimmed.to_owned();
    }
    trimmed.chars().take(MAX).collect()
}

/// Splits a response into the records that need retrying and why.
///
/// Response records correlate with request records by position, which the API
/// guarantees, so the pairing is a zip. Kept as a free function because it is the piece
/// with the logic in it, and testing it needs no network.
///
/// A response whose length does not match the request cannot be read positionally, so
/// everything is retried rather than guessed at: a duplicate is recoverable and a lost
/// row is not. A record counts as succeeded when it has no error code -- checking for a
/// sequence number instead would treat a malformed success as a failure.
fn failed_entries(
    sent: Vec<PutRecordsRequestEntry>,
    results: &[PutRecordsResultEntry],
) -> (Vec<PutRecordsRequestEntry>, Vec<String>) {
    if results.len() != sent.len() {
        let reason = format!(
            "response had {} results for {} records, so they could not be matched \
             positionally",
            results.len(),
            sent.len()
        );
        return (sent, vec![reason]);
    }

    let mut retry = Vec::new();
    let mut reasons = Vec::new();
    for (entry, result) in sent.into_iter().zip(results) {
        if let Some(code) = result.error_code() {
            reasons.push(code.to_owned());
            retry.push(entry);
        }
    }
    (retry, reasons)
}

/// Counts each distinct reason, so a log line does not repeat one 500 times.
fn summarise(reasons: &[String]) -> String {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for reason in reasons {
        *counts.entry(reason.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(reason, count)| format!("{reason} x{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Waits before a retry, doubling each time.
///
/// Throttling is the expected reason a record is rejected, and retrying immediately
/// would make it worse.
async fn backoff(attempt: usize) {
    let millis = 50u64 << (attempt.min(6) as u32);
    tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::blob_store::LocalBlobStore;

    fn batch(request_id: &str, rows: usize) -> InferenceLogBatch {
        InferenceLogBatch {
            request_id: request_id.to_owned(),
            model_id: "m".to_owned(),
            inference_log_rows: (0..rows)
                .map(|i| hushar::hushar_proto::InferenceLogRow {
                    row_id: format!("row-{i}"),
                    features: Default::default(),
                    scores: vec![0.5],
                })
                .collect(),
        }
    }

    fn entry(key: &str) -> PutRecordsRequestEntry {
        PutRecordsRequestEntry::builder()
            .data(aws_sdk_kinesis::primitives::Blob::new(b"x".to_vec()))
            .partition_key(key)
            .build()
            .expect("valid entry")
    }

    fn ok_result() -> PutRecordsResultEntry {
        PutRecordsResultEntry::builder()
            .shard_id("shard-1")
            .sequence_number("42")
            .build()
    }

    fn failed_result(code: &str) -> PutRecordsResultEntry {
        PutRecordsResultEntry::builder()
            .error_code(code)
            .error_message("throttled")
            .build()
    }

    // ------------------------------------------------------- destinations

    #[test]
    fn a_kinesis_uri_is_a_stream_name() {
        assert_eq!(
            DataDestination::parse("kinesis://inference-logs").expect("valid"),
            DataDestination::Kinesis {
                stream: "inference-logs".into()
            }
        );
        assert_eq!(
            DataDestination::parse("kinesis://inference-logs/").expect("valid"),
            DataDestination::Kinesis {
                stream: "inference-logs".into()
            }
        );
    }
    /// `kinesis://bucket/prefix` is what someone writes after using s3://, and silently
    /// taking "bucket" as the stream would send logs somewhere real but wrong.
    #[test]
    fn a_kinesis_uri_with_a_path_is_refused_rather_than_mangled() {
        let err = DataDestination::parse("kinesis://stream/extra/path")
            .expect_err("a path is not a stream name");
        assert!(
            err.to_string().contains("stream name"),
            "the error should explain the shape: {err}"
        );
        assert!(DataDestination::parse("kinesis://").is_err());
    }
    /// And an unsupported scheme is still reported by `Location`.
    #[test]
    fn other_uris_still_mean_a_blob_store() {
        assert!(matches!(
            DataDestination::parse("s3://bucket/logs").expect("valid"),
            DataDestination::Blob(Location::S3 { .. })
        ));
        assert!(matches!(
            DataDestination::parse("/tmp/logs").expect("valid"),
            DataDestination::Blob(Location::Local(_))
        ));
        assert!(DataDestination::parse("az://container/logs").is_err());
    }

    #[test]
    fn a_destination_round_trips_through_its_uri() {
        for uri in ["kinesis://my-stream", "s3://bucket/logs", "/tmp/logs"] {
            assert_eq!(DataDestination::parse(uri).expect("valid").uri(), uri);
        }
    }

    // ---------------------------------------------------------- sampling

    /// The default has to be what the service did before there was a lever.
    #[test]
    fn a_rate_of_one_keeps_every_request() {
        let sampler = Sampler::new(1.0);
        assert!((0..100).all(|_| sampler.sample()));
        assert_eq!(sampler.describe(), "every request");
    }

    #[test]
    fn a_rate_of_zero_keeps_nothing() {
        let sampler = Sampler::new(0.0);
        assert!(!(0..100).any(|_| sampler.sample()));
        assert_eq!(sampler.describe(), "no requests");
    }
    /// Counted rather than randomised, so the fraction is exact over any window rather
    /// than approached over a long one -- which is what makes a small sample rate
    /// predictable instead of a gamble.
    #[test]
    fn a_fractional_rate_keeps_exactly_that_fraction() {
        let sampler = Sampler::new(0.1);
        let kept = (0..1000).filter(|_| sampler.sample()).count();
        assert_eq!(kept, 100, "one in ten, not approximately one in ten");
    }
    /// A rate that is not a unit fraction cannot be realised exactly by counting, so
    /// the banner has to report what was chosen rather than what was asked for.
    #[test]
    fn a_rate_between_unit_fractions_reports_what_it_settled_on() {
        let sampler = Sampler::new(0.15);
        assert_eq!(sampler.describe(), "1 request in 7 (14.3%)");

        let kept = (0..700).filter(|_| sampler.sample()).count();
        assert_eq!(kept, 100);
    }
    /// Out-of-range and non-finite values are refused by the command line, but a
    /// sampler that divided by NaN would produce a nonsense interval rather than a
    /// clear one, so it is handled here too.
    #[test]
    fn a_nonsensical_rate_degrades_to_keeping_nothing() {
        for rate in [f64::NAN, -1.0, -0.0] {
            assert!(
                !Sampler::new(rate).sample(),
                "{rate} should not silently keep everything"
            );
        }
        assert!(Sampler::new(2.0).sample(), "above one is still everything");
    }
    /// Requests arrive on every worker at once, so the count has to be shared: two
    /// samplers' worth of state would keep twice the configured fraction.
    #[tokio::test]
    async fn sampling_is_counted_across_concurrent_requests() {
        let sampler = Arc::new(Sampler::new(0.25));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let sampler = Arc::clone(&sampler);
            handles.push(tokio::spawn(async move {
                (0..50).filter(|_| sampler.sample()).count()
            }));
        }
        let mut kept = 0;
        for handle in handles {
            kept += handle.await.expect("sampler task");
        }
        assert_eq!(kept, 100, "one in four of 400 requests");
    }

    // -------------------------------------------------- partial failures
    /// The behaviour that makes a Kinesis producer correct. `PutRecords` answers 200
    /// with per-record errors, so a producer that only checks the `Result` reports
    /// success while losing rows.
    #[test]
    fn only_the_rejected_records_are_retried() {
        let sent = vec![entry("a"), entry("b"), entry("c")];
        let results = vec![
            ok_result(),
            failed_result("ProvisionedThroughputExceededException"),
            ok_result(),
        ];

        let (retry, reasons) = failed_entries(sent, &results);
        assert_eq!(retry.len(), 1, "only the middle record failed");
        assert_eq!(retry[0].partition_key(), "b", "and it must be that one");
        assert_eq!(reasons, vec!["ProvisionedThroughputExceededException"]);
    }

    #[test]
    fn a_fully_successful_response_retries_nothing() {
        let sent = vec![entry("a"), entry("b")];
        let results = vec![ok_result(), ok_result()];
        let (retry, reasons) = failed_entries(sent, &results);
        assert!(retry.is_empty());
        assert!(reasons.is_empty());
    }

    #[test]
    fn a_fully_failed_response_retries_everything_in_order() {
        let sent = vec![entry("a"), entry("b"), entry("c")];
        let results = vec![
            failed_result("InternalFailure"),
            failed_result("ProvisionedThroughputExceededException"),
            failed_result("InternalFailure"),
        ];
        let (retry, reasons) = failed_entries(sent, &results);
        let keys: Vec<&str> = retry.iter().map(|e| e.partition_key()).collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
        assert_eq!(reasons.len(), 3);
    }
    /// Positional matching is only valid when the lengths agree. Guessing would risk
    /// dropping a record that actually failed; a duplicate is recoverable and a lost
    /// row is not.
    #[test]
    fn a_response_that_does_not_line_up_retries_everything() {
        let sent = vec![entry("a"), entry("b"), entry("c")];
        let results = vec![ok_result()];
        let (retry, reasons) = failed_entries(sent, &results);
        assert_eq!(retry.len(), 3, "all three should be retried");
        assert_eq!(reasons.len(), 1, "with one explanation of why");
        assert!(
            reasons[0].contains("could not be matched"),
            "got {reasons:?}"
        );
    }
    /// Five hundred identical throttle messages in one log line is not a log line.
    #[test]
    fn reasons_are_counted_not_repeated() {
        let reasons = vec![
            "ProvisionedThroughputExceededException".to_owned(),
            "ProvisionedThroughputExceededException".to_owned(),
            "InternalFailure".to_owned(),
        ];
        let summary = summarise(&reasons);
        assert!(
            summary.contains("ProvisionedThroughputExceededException x2"),
            "got {summary}"
        );
        assert!(summary.contains("InternalFailure x1"), "got {summary}");
    }

    // ----------------------------------------------------------- records

    #[test]
    fn a_batch_becomes_a_record_keyed_by_its_request_id() {
        let entry = to_entry(&batch("req-7", 3)).expect("valid record");
        assert_eq!(entry.partition_key(), "req-7");
        assert!(
            !entry.data().as_ref().is_empty(),
            "the encoded batch should be the payload"
        );
    }
    /// Kinesis rejects an empty partition key, and it rejects the whole request rather
    /// than that record, so this cannot be passed through. Random rather than constant,
    /// or every keyless record would land on one shard.
    #[test]
    fn an_absent_request_id_still_produces_a_usable_key() {
        let a = partition_key("");
        let b = partition_key("   ");
        assert!(!a.is_empty() && !b.is_empty());
        assert_ne!(a, b, "keyless records should still spread across shards");
    }
    /// On a character boundary: the limit is characters, and a byte slice could split a
    /// multi-byte one.
    #[test]
    fn an_over_long_request_id_is_truncated_to_the_limit() {
        let key = partition_key(&"x".repeat(1000));
        assert_eq!(key.chars().count(), 256);

        let key = partition_key(&"é".repeat(1000));
        assert_eq!(key.chars().count(), 256);
        assert!(key.is_char_boundary(key.len()));
    }
    /// Sending it would fail the whole `PutRecords` call and take 499 good records with
    /// it, so it is refused individually. Comfortably past the per-record limit once
    /// encoded.
    #[test]
    fn a_record_over_the_size_limit_is_reported_not_sent() {
        let huge = InferenceLogBatch {
            request_id: "big".into(),
            model_id: "m".into(),
            inference_log_rows: vec![hushar::hushar_proto::InferenceLogRow {
                row_id: "r".into(),
                features: Default::default(),
                scores: vec![0.0; KINESIS_MAX_RECORD_BYTES],
            }],
        };
        let err = to_entry(&huge).expect_err("over the limit");
        assert!(err.contains("per-record limit"), "got {err}");
    }

    // -------------------------------------------------------- blob sink

    /// A blob sink writing to a fresh temporary directory.
    fn blob_sink(dir: &std::path::Path, capacity: usize) -> BlobDataSink {
        BlobDataSink::new(
            Arc::new(LocalBlobStore) as Arc<dyn BlobStore>,
            Location::Local(dir.to_path_buf()),
            capacity,
            2,
        )
    }

    fn temp_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("hushar-datasink-{}", uuid::Uuid::new_v4()))
    }

    /// The write is spawned, so `flush` is what makes it observable -- it waits for the
    /// sends in the air, which is the property shutdown depends on.
    #[tokio::test]
    async fn the_blob_sink_writes_once_it_reaches_capacity() {
        let dir = temp_dir();
        let sink = blob_sink(&dir, 2);

        sink.record(batch("a", 1));
        assert_eq!(walk(&dir).len(), 0, "one of two: still buffered");

        sink.record(batch("b", 1));
        sink.flush().await;
        assert_eq!(
            walk(&dir).len(),
            1,
            "capacity reached, so one object written"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
    /// One batch with a capacity of 100: without the flush this writes nothing.
    #[tokio::test]
    async fn the_blob_sink_does_not_lose_a_partial_buffer_on_flush() {
        let dir = temp_dir();
        let sink = blob_sink(&dir, 100);
        sink.record(batch("only", 2));
        sink.flush().await;
        assert_eq!(walk(&dir).len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
    /// The service can stop before any batch arrives, and an empty object would be
    /// noise for whatever reads the prefix.
    #[tokio::test]
    async fn flushing_an_empty_blob_sink_writes_nothing() {
        let dir = temp_dir();
        let sink = blob_sink(&dir, 10);
        sink.flush().await;
        assert_eq!(walk(&dir).len(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }
    /// The point of the whole change: recording must not wait on the destination. A
    /// local write is fast, but "faster than a hundred writes" is the shape of the
    /// claim, and an awaited put could not satisfy it.
    #[tokio::test]
    async fn recording_returns_without_waiting_for_the_write() {
        let dir = temp_dir();
        let sink = blob_sink(&dir, 1);

        let start = std::time::Instant::now();
        for i in 0..100 {
            sink.record(batch(&format!("r{i}"), 1));
        }
        let recording = start.elapsed();
        sink.flush().await;

        assert!(
            recording < std::time::Duration::from_millis(50),
            "100 records took {recording:?}, so recording is waiting on the write"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
    /// Zero-padded, so lexical order matches chronological order, which is what makes a
    /// partition listing usable.
    #[test]
    fn a_log_key_is_partitioned_by_time_and_unique() {
        let a = time_partitioned_key();
        let b = time_partitioned_key();
        assert_ne!(a, b, "two keys in the same minute must not collide");
        for part in ["year=", "month=", "day=", "hour=", "mi="] {
            assert!(a.contains(part), "{a} should contain {part}");
        }
        let month = a
            .split("month=")
            .nth(1)
            .expect("month")
            .split('/')
            .next()
            .expect("value");
        assert_eq!(month.len(), 2, "month should be zero-padded, got {month}");
    }

    // ----------------------------------------------------- kinesis sink

    /// A Kinesis sink with credentials that will never work.
    ///
    /// Enough to test buffering, which is the part with logic in it. Sending is
    /// covered by `failed_entries` and `to_entry`, which need no network.
    fn offline_kinesis(capacity: usize) -> KinesisDataSink {
        let config = aws_sdk_kinesis::Config::builder()
            .behavior_version_latest()
            .region(aws_sdk_kinesis::config::Region::new("us-east-1"))
            .credentials_provider(aws_sdk_kinesis::config::Credentials::for_tests())
            .build();
        KinesisDataSink::new(
            Arc::new(aws_sdk_kinesis::Client::from_conf(config)),
            "test-stream",
            capacity,
            2,
        )
    }

    #[tokio::test]
    async fn the_kinesis_sink_buffers_up_to_capacity() {
        let sink = offline_kinesis(3);
        sink.record(batch("a", 1));
        sink.record(batch("b", 1));
        assert_eq!(sink.accumulator.buffered(), 2, "still buffered");
    }
    /// 500 records per call is a hard API limit; a larger buffer would build a request
    /// the service rejects whole. And zero would send one record per call, which is the
    /// expensive way. Reported by `batch_size`, so a clamped value is visible at
    /// startup rather than only in the bill.
    #[tokio::test]
    async fn the_kinesis_capacity_cannot_exceed_what_put_records_accepts() {
        assert_eq!(offline_kinesis(50_000).batch_size(), KINESIS_MAX_RECORDS);
        assert_eq!(offline_kinesis(0).batch_size(), 1);
    }
    /// A blob store takes an object of any size, so this is the sink where a large
    /// batch is purely a memory trade and nothing reduces it.
    #[tokio::test]
    async fn a_blob_sink_keeps_whatever_batch_size_it_was_given() {
        let dir = temp_dir();
        assert_eq!(blob_sink(&dir, 50_000).batch_size(), 50_000);
    }
    /// Consistent with `BlobStore::backend` and `MetricsSink::backend`: the backend,
    /// not the address. The address comes from `DataDestination::uri`, so a banner that
    /// printed both would not repeat itself.
    #[tokio::test]
    async fn each_sink_names_its_backend_the_way_its_siblings_do() {
        assert_eq!(offline_kinesis(10).backend(), "Amazon Kinesis");
        let dir = temp_dir();
        assert_eq!(blob_sink(&dir, 1).backend(), "local filesystem");
    }
    /// Throttling is the expected rejection, so retrying at the same rate would make it
    /// worse. Timed rather than inspected, since the delay is the point.
    #[tokio::test]
    async fn backoff_grows_with_each_attempt() {
        let start = std::time::Instant::now();
        backoff(1).await;
        let first = start.elapsed();

        let start = std::time::Instant::now();
        backoff(3).await;
        let third = start.elapsed();

        assert!(
            third > first,
            "attempt 3 should wait longer than attempt 1 ({third:?} vs {first:?})"
        );
    }

    /// Every file beneath `dir`, recursively.
    fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut found = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return found;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                found.extend(walk(&path));
            } else {
                found.push(path);
            }
        }
        found
    }
}
