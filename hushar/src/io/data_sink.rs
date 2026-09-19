// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Where the data a request produces goes.
//!
//! Today that is the inference log: the features as received and the scores they
//! produced, as batched protobuf objects in a blob store or as records on a Kinesis
//! data stream.
//!
//! The request path calls [`DataSink::record`] directly, which appends to an
//! [`Accumulator`](super::accumulator::Accumulator) and returns. Nothing awaits a
//! network call on a request, and no thread is reserved away from serving.
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
/// # Fields
///
/// * `Blob` - batched protobuf objects beneath a prefix
/// * `Kinesis` - records on a Kinesis data stream
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
    /// filesystem paths are both accepted.
    ///
    /// # Errors
    ///
    /// If a `kinesis://` URI carries no stream name or carries a path, or if
    /// [`Location::parse`] rejects the URI.
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
/// The decision is made before scoring, so an unsampled request never clones its
/// feature maps. See [`FeatureLogging`](crate::inference::scoring::FeatureLogging).
///
/// **Sampling is per request, not per row.** A request that is kept is kept whole, so
/// a consumer joining rows back to one `request_id` never sees a partial batch.
///
/// Counted rather than randomised, so exactly the configured fraction is kept over any
/// window and no random-number generator sits on the request path.
#[derive(Debug)]
pub(crate) struct Sampler {
    /// Keep one request in this many. Zero keeps none.
    every: u64,
    seen: AtomicU64,
}

impl Sampler {
    /// Builds a sampler that keeps `rate` of requests.
    ///
    /// Realised as one request in `n`, so a rate that is not a unit fraction becomes
    /// the nearest one: a rate of 0.15 is kept as one in 7, which is 14.3%.
    /// [`Self::describe`] reports what was chosen rather than what was asked for.
    ///
    /// # Arguments
    ///
    /// * `rate` - the fraction of requests to record, in `0.0..=1.0`. NaN and anything
    ///   at or below zero keep nothing; anything at or above one keeps everything.
    pub(crate) fn new(rate: f64) -> Self {
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
    /// Safe to call concurrently; the count is shared across all callers.
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
/// method takes `&self`. Batching belongs here, so that each destination batches the
/// way its own API wants.
#[tonic::async_trait]
pub(crate) trait DataSink: Debug + Send + Sync {
    /// Which destination this is, for the startup banner.
    fn backend(&self) -> &str;

    /// Batches accumulated before one send, after any clamping the API forces.
    ///
    /// Reported at startup, so a value the destination's own limit reduced is visible.
    fn batch_size(&self) -> usize;

    /// Accepts one batch, on the request path.
    ///
    /// Synchronous and non-blocking: it appends to a buffer, and on the request that
    /// fills it spawns the send. No request ever awaits a network call.
    ///
    /// # Panics
    ///
    /// If called outside a tokio runtime, since the send is spawned.
    fn record(&self, batch: InferenceLogBatch);

    /// Sends anything buffered and waits for the sends already in the air.
    ///
    /// Called once at shutdown, before the runtime is dropped: a partial buffer and
    /// every detached send would otherwise go with it.
    async fn flush(&self);
}

/// The part of a blob sink that talks to the store.
#[derive(Debug)]
struct BlobWriter {
    store: Arc<dyn BlobStore>,
    prefix: Location,
}

impl BlobWriter {
    /// Encodes and writes one object, reporting a failure on stderr rather than
    /// dropping it silently.
    ///
    /// An empty `batches` writes nothing.
    async fn write(&self, batches: Vec<InferenceLogBatch>) {
        if batches.is_empty() {
            return;
        }
        let rows: usize = batches.iter().map(|b| b.row_ids.len()).sum();
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
/// listing everything.
#[derive(Debug)]
pub(crate) struct BlobDataSink {
    writer: Arc<BlobWriter>,
    accumulator: Accumulator<InferenceLogBatch>,
}

impl BlobDataSink {
    /// Builds a blob sink writing beneath `prefix`.
    ///
    /// # Arguments
    ///
    /// * `store` - the blob store the objects are written to
    /// * `prefix` - the location every object is written beneath
    /// * `capacity` - batches per object. Not clamped, since an object store takes an
    ///   object of any size, so this is purely a memory decision.
    /// * `max_in_flight` - writes allowed at once
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
    /// The store's own name, such as `Amazon S3` or `local filesystem`.
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
/// Hive-style partitions, zero-padded so lexical order matches chronological order, and
/// a UUID so concurrent writes never collide on a name.
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
/// Bounded, so a throttled stream cannot hold a send slot indefinitely.
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
    /// `PutRecords` answers **200 even when individual records were rejected**,
    /// reporting them positionally in the response body, so the body is inspected and
    /// only the rejected records are retried. A whole-request failure leaves every
    /// record pending. Anything still pending after `KINESIS_MAX_ATTEMPTS` is reported
    /// as lost on stderr, as is a batch too large to encode into one record.
    async fn send(&self, batches: Vec<InferenceLogBatch>) {
        let mut pending: Vec<PutRecordsRequestEntry> = Vec::with_capacity(batches.len());
        for batch in batches {
            match to_entry(&batch) {
                Ok(entry) => pending.push(entry),
                Err(reason) => eprintln!(
                    "logs: dropped request {:?} with {} rows: {reason}",
                    batch.request_id,
                    batch.row_ids.len()
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
    /// Builds a Kinesis sink for `stream`.
    ///
    /// # Arguments
    ///
    /// * `client` - the Kinesis client the records are sent with
    /// * `stream` - the data stream's name
    /// * `capacity` - batches per send, clamped to `KINESIS_MAX_RECORDS`, the 500 records
    ///   `PutRecords` accepts in one call, because one batch becomes one record
    /// * `max_in_flight` - sends allowed at once
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
/// The partition key is the batch's `request_id`, which Kinesis hashes to pick a shard:
/// an even spread, with one request's rows kept together for a consumer.
///
/// # Errors
///
/// If the encoded batch and its partition key together exceed
/// `KINESIS_MAX_RECORD_BYTES`, or the record cannot be built.
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
/// The service requires 1 to 256 characters, so a blank `request_id` becomes a random
/// UUID and an over-long one is truncated to 256 characters, on a character boundary.
/// Either would otherwise be rejected for the whole request, not just that record.
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
/// guarantees, so the pairing is a zip. A record counts as succeeded when it carries no
/// error code.
///
/// # Returns
///
/// The entries to retry, and the error code seen for each. A response whose length does
/// not match the request cannot be read positionally, so every entry is retried with one
/// reason explaining why.
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

/// Counts each distinct reason, so one line does not repeat a code once per record.
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

/// Waits before a retry, doubling with each attempt and levelling off after the sixth.
async fn backoff(attempt: usize) {
    let millis = 50u64 << (attempt.min(6) as u32);
    tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::blob_store::LocalBlobStore;
    use hushar::hushar_proto::FloatArray;

    /// Scores in the shape a response carries them: one buffer, single precision.
    fn score_type(values: Vec<f32>) -> hushar::hushar_proto::ScoreType {
        hushar::hushar_proto::ScoreType {
            score_type: Some(hushar::hushar_proto::score_type::ScoreType::FloatScores(
                FloatArray { values },
            )),
        }
    }

    /// A batch of `rows` rows: `row_ids` states the row count, one score per row.
    fn batch(request_id: &str, rows: usize) -> InferenceLogBatch {
        InferenceLogBatch {
            request_id: request_id.to_owned(),
            model_id: "m".to_owned(),
            row_ids: row_ids(rows),
            features: text_column(rows),
            scores: Some(score_type(vec![0.5; rows])),
        }
    }

    /// The ids a log batch states its rows with.
    fn row_ids(rows: usize) -> Vec<String> {
        (0..rows).map(|i| format!("row-{i}")).collect()
    }

    /// One text feature, so a batch carries what was received as well as what it scored.
    fn text_column(
        rows: usize,
    ) -> std::collections::HashMap<String, hushar::hushar_proto::Feature> {
        use crate::inference::batch::FeatureData;
        use crate::inference::proto_columns::test_support;

        test_support::one(
            "city",
            FeatureData::Str((0..rows).map(|i| format!("city-{i}")).collect()),
        )
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
    /// A `kinesis://` URI carrying a path is refused rather than read as a stream name.
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
    /// Other schemes parse as blob-store locations, and an unsupported one errors.
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

    /// A rate of one keeps every request.
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
    /// A fractional rate keeps exactly that fraction, not approximately.
    #[test]
    fn a_fractional_rate_keeps_exactly_that_fraction() {
        let sampler = Sampler::new(0.1);
        let kept = (0..1000).filter(|_| sampler.sample()).count();
        assert_eq!(kept, 100, "one in ten, not approximately one in ten");
    }
    /// A rate between unit fractions describes the interval it settled on.
    #[test]
    fn a_rate_between_unit_fractions_reports_what_it_settled_on() {
        let sampler = Sampler::new(0.15);
        assert_eq!(sampler.describe(), "1 request in 7 (14.3%)");

        let kept = (0..700).filter(|_| sampler.sample()).count();
        assert_eq!(kept, 100);
    }
    /// NaN and negative rates keep nothing; a rate above one keeps everything.
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
    /// The kept fraction is exact when requests arrive on many tasks at once.
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

    /// Only the records the response reported an error for are retried.
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
    /// A response whose length does not match the request retries every record.
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
    /// Repeated error codes are summarised with a count rather than listed.
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

    #[test]
    fn a_batch_becomes_a_record_keyed_by_its_request_id() {
        let entry = to_entry(&batch("req-7", 3)).expect("valid record");
        assert_eq!(entry.partition_key(), "req-7");
        assert!(
            !entry.data().as_ref().is_empty(),
            "the encoded batch should be the payload"
        );
    }
    /// A blank request id becomes a non-empty key, and a different one each time.
    #[test]
    fn an_absent_request_id_still_produces_a_usable_key() {
        let a = partition_key("");
        let b = partition_key("   ");
        assert!(!a.is_empty() && !b.is_empty());
        assert_ne!(a, b, "keyless records should still spread across shards");
    }
    /// An over-long request id is truncated to the limit, on a character boundary.
    #[test]
    fn an_over_long_request_id_is_truncated_to_the_limit() {
        let key = partition_key(&"x".repeat(1000));
        assert_eq!(key.chars().count(), 256);

        let key = partition_key(&"é".repeat(1000));
        assert_eq!(key.chars().count(), 256);
        assert!(key.is_char_boundary(key.len()));
    }
    /// A batch over the per-record byte limit is reported as an error, not sent.
    #[test]
    fn a_record_over_the_size_limit_is_reported_not_sent() {
        let huge = InferenceLogBatch {
            request_id: "big".into(),
            model_id: "m".into(),
            row_ids: row_ids(1),
            features: text_column(1),
            scores: Some(score_type(vec![0.0; KINESIS_MAX_RECORD_BYTES])),
        };
        let err = to_entry(&huge).expect_err("over the limit");
        assert!(err.contains("per-record limit"), "got {err}");
    }

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

    /// One object is written once the buffer reaches capacity, observable after a flush.
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
    /// A flush writes a buffer that never reached capacity.
    #[tokio::test]
    async fn the_blob_sink_does_not_lose_a_partial_buffer_on_flush() {
        let dir = temp_dir();
        let sink = blob_sink(&dir, 100);
        sink.record(batch("only", 2));
        sink.flush().await;
        assert_eq!(walk(&dir).len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
    /// Flushing with nothing buffered writes no object.
    #[tokio::test]
    async fn flushing_an_empty_blob_sink_writes_nothing() {
        let dir = temp_dir();
        let sink = blob_sink(&dir, 10);
        sink.flush().await;
        assert_eq!(walk(&dir).len(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }
    /// Recording returns without waiting for the write.
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
    /// A log key carries zero-padded time partitions and is unique per call.
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

    /// A Kinesis sink with credentials that will never work, for testing buffering.
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
    /// The Kinesis batch size is clamped to what `PutRecords` accepts, and to at least
    /// one.
    #[tokio::test]
    async fn the_kinesis_capacity_cannot_exceed_what_put_records_accepts() {
        assert_eq!(offline_kinesis(50_000).batch_size(), KINESIS_MAX_RECORDS);
        assert_eq!(offline_kinesis(0).batch_size(), 1);
    }
    /// A blob sink reports the batch size it was given, unclamped.
    #[tokio::test]
    async fn a_blob_sink_keeps_whatever_batch_size_it_was_given() {
        let dir = temp_dir();
        assert_eq!(blob_sink(&dir, 50_000).batch_size(), 50_000);
    }
    /// Each sink names its backend rather than its address.
    #[tokio::test]
    async fn each_sink_names_its_backend_the_way_its_siblings_do() {
        assert_eq!(offline_kinesis(10).backend(), "Amazon Kinesis");
        let dir = temp_dir();
        assert_eq!(blob_sink(&dir, 1).backend(), "local filesystem");
    }
    /// A later attempt waits longer than an earlier one.
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
