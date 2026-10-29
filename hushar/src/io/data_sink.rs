// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Where the data a request produces goes.
//!
//! Today that is the inference log: the features as received and the scores they
//! produced, as batched protobuf objects in a blob store.
//!
//! The request path calls [`DataSink::record`] directly, which appends to an
//! [`Accumulator`](super::accumulator::Accumulator) and returns. Nothing awaits a
//! network call on a request, and no thread is reserved away from serving.
//!
//! # Adding a destination
//!
//! 1. Implement [`DataSink`].
//! 2. Build it in place of [`BlobDataSink`] in `build_data_sink` in `main`.

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{Datelike, Timelike};
use hushar::hushar_proto::{InferenceLogBatch, InferenceLogs};
use prost::Message;

use crate::io::accumulator::Accumulator;
use crate::io::{BlobStore, Location};

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
    shards: u32,
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
        let target = self.prefix.join(&partitioned_key(self.shards));
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
/// The default. Objects are named by shard and time so they can be queried by partition
/// without listing everything.
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
    /// * `shards` - how many key prefixes to spread objects over, at least one. See
    ///   [`partitioned_key`]
    /// * `capacity` - batches per object. Not clamped, since an object store takes an
    ///   object of any size, so this is purely a memory decision.
    /// * `max_in_flight` - writes allowed at once
    pub(crate) fn new(
        store: Arc<dyn BlobStore>,
        prefix: Location,
        shards: u32,
        capacity: usize,
        max_in_flight: usize,
    ) -> Self {
        Self {
            writer: Arc::new(BlobWriter {
                store,
                prefix,
                shards,
            }),
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

/// A sharded, time-partitioned object name, unique per write.
///
/// A `shard=` segment leads, then Hive-style time partitions, zero-padded so lexical
/// order matches chronological order, then a UUID so concurrent writes never collide on
/// a name.
///
/// # The leading shard
///
/// Amazon S3 scales its request rate per key prefix, and splits a hot prefix only after
/// observing the load. A fleet writing beneath one `mi=` prefix shares one rate limit
/// while that split happens, so writes are spread over `shards` distinct prefixes
/// instead. It leads because a prefix is matched from the left: entropy any later —
/// the object's own UUID, for instance — shares a prefix with everything else written
/// in the same minute.
///
/// Spelled `shard=07` rather than `07` because Hive-style partition discovery stops at
/// the first level that is not `key=value`, and a bare segment would take the time
/// partitions below it out of the table with it.
///
/// # Arguments
///
/// * `shards` - prefixes to spread over, at least one. The number is taken from the
///   same UUID that names the object, and padded to a fixed width so that no shard's
///   prefix is a prefix of another's.
///
/// # Panics
///
/// If `shards` is zero, which names no prefix to write beneath.
/// [`validate`](crate::config::HusharServiceConfig::validate) refuses it at startup.
fn partitioned_key(shards: u32) -> String {
    let now = chrono::Utc::now();
    let id = uuid::Uuid::new_v4();
    let shard = (id.as_u128() % u128::from(shards)) as u32;
    let width = (shards - 1).to_string().len();
    format!(
        "shard={shard:0width$}/year={}/month={:02}/day={:02}/hour={:02}/mi={:02}/{id}_{}.pb",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute(),
        now.timestamp_micros()
    )
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

    /// A blob sink writing to a fresh temporary directory.
    fn blob_sink(dir: &std::path::Path, capacity: usize) -> BlobDataSink {
        BlobDataSink::new(
            Arc::new(LocalBlobStore) as Arc<dyn BlobStore>,
            Location::Local(dir.to_path_buf()),
            1,
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
    /// The value of one `key=value` segment of a log key.
    fn segment(key: &str, field: &str) -> String {
        key.split(field)
            .nth(1)
            .expect(field)
            .split('/')
            .next()
            .expect("value")
            .to_owned()
    }

    /// A log key leads with a shard, then carries zero-padded time partitions, and is
    /// unique per call.
    #[test]
    fn a_log_key_leads_with_a_shard_then_partitions_by_time() {
        let a = partitioned_key(16);
        let b = partitioned_key(16);
        assert_ne!(a, b, "two keys in the same minute must not collide");
        assert!(a.starts_with("shard="), "the shard must lead: {a}");
        for part in ["year=", "month=", "day=", "hour=", "mi="] {
            assert!(a.contains(part), "{a} should contain {part}");
        }
        let month = segment(&a, "month=");
        assert_eq!(month.len(), 2, "month should be zero-padded, got {month}");
    }
    /// Shard numbers stay inside the configured count, and over many keys use all of it.
    #[test]
    fn shard_numbers_cover_the_configured_range() {
        let seen: std::collections::BTreeSet<String> = (0..2_000)
            .map(|_| segment(&partitioned_key(8), "shard="))
            .collect();
        assert_eq!(
            seen,
            (0..8).map(|n| n.to_string()).collect(),
            "all eight shards should be used, and nothing outside them"
        );
    }
    /// Shard numbers are padded to a fixed width, so no shard's prefix contains
    /// another's and S3 has one byte position to split on.
    #[test]
    fn a_shard_number_is_padded_to_a_fixed_width() {
        let widths: std::collections::BTreeSet<usize> = (0..500)
            .map(|_| segment(&partitioned_key(1_000), "shard=").len())
            .collect();
        assert_eq!(
            widths,
            std::collections::BTreeSet::from([3]),
            "1000 shards should all be three digits"
        );
    }
    /// One shard still names itself, so the layout does not change shape with the count.
    #[test]
    fn a_single_shard_still_names_itself() {
        assert!(partitioned_key(1).starts_with("shard=0/"));
    }
    /// The shard reaches the written object, not just the key the writer computed.
    #[tokio::test]
    async fn a_written_object_sits_beneath_its_shard() {
        let dir = temp_dir();
        let sink = BlobDataSink::new(
            Arc::new(LocalBlobStore) as Arc<dyn BlobStore>,
            Location::Local(dir.clone()),
            4,
            1,
            2,
        );
        sink.record(batch("a", 1));
        sink.flush().await;

        let written = walk(&dir);
        assert_eq!(written.len(), 1, "one batch at capacity one");
        let path = written[0].display().to_string();
        assert!(
            path.contains("/shard="),
            "{path} should sit beneath a shard"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
    /// A blob sink reports the batch size it was given, unclamped.
    #[tokio::test]
    async fn a_blob_sink_keeps_whatever_batch_size_it_was_given() {
        let dir = temp_dir();
        assert_eq!(blob_sink(&dir, 50_000).batch_size(), 50_000);
    }
    /// A sink names its store rather than its address.
    #[tokio::test]
    async fn a_sink_names_its_store_not_its_address() {
        let dir = temp_dir();
        assert_eq!(blob_sink(&dir, 1).backend(), "local filesystem");
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
