// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Merging the rows of concurrent requests into one engine call.
//!
//! The opposite of cutting a request into mini batches, and the answer for a device
//! already filled by one call. A call has a large fixed cost regardless of its rows, so
//! filling it spreads that cost over more rows; splitting a request divides a call and
//! pays the cost again per mini batch.
//!
//! ```text
//!   merge { max_rows: 256, max_delay_micros: 300 }
//!
//!   request A, 100 rows ─┐
//!   request B,  80 rows ─┼─► one call of 256 rows ─► scatter each caller its own rows
//!   request C,  40 rows ─┘      76 rows of ballast
//! ```
//!
//! Every call is padded to exactly `max_rows`, so the engine is always shown one shape.
//! A provider that compiles per shape would otherwise build an engine for every width it
//! happened to see, and those builds take seconds.
//!
//! A request larger than `max_rows` spans consecutive calls. That is the cap doing its
//! job, not mini batching: the rows still arrive in one call each as full as it can be.

use std::ops::Range;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use crate::inference::RequestReport;
use crate::inference::backend::InferenceBackend;
use crate::inference::batch::ScoreData;
use crate::inference::input_builder::{InputBuilder, RequestColumns};
use crate::inference::scoring::ScoredBatch;
use crate::inference::session_pool::SessionPool;

/// The pool of sessions a merged call leases from.
type Sessions = SessionPool<Arc<dyn InferenceBackend>>;

/// What a caller gets back for one of its merged calls.
///
/// The error is a `String` rather than an [`crate::inference::InferenceError`] because
/// one failed call fails every request in it, and a boxed error cannot be handed to
/// several recipients.
pub(crate) type Reply = Result<ScoredBatch, String>;

/// One request's rows, waiting for a call to carry them.
struct Pending {
    columns: Arc<RequestColumns>,
    rows: Range<usize>,
    reply: oneshot::Sender<Reply>,
    /// Entries queued when this one joined, itself included.
    queue_depth: usize,
}

impl Pending {
    /// Rows this entry contributes to a call.
    fn len(&self) -> usize {
        self.rows.end.saturating_sub(self.rows.start)
    }
}

/// Queues requests and runs them as merged calls.
///
/// One queue per model, drained by one thread per request lane. Each thread leases a
/// session for the call it is running, so the lanes are as independent as they are for an
/// unmerged request and no session is left loaded but unused.
#[derive(Debug)]
pub(crate) struct Merger {
    queue: mpsc::Sender<Pending>,
    max_rows: usize,
    /// Entries queued and not yet taken by a lane, for the `QueueDepth` measure.
    waiting: Arc<std::sync::atomic::AtomicUsize>,
}

impl Merger {
    /// Starts the merger's threads, which run until the merger is dropped.
    ///
    /// # Arguments
    ///
    /// * `sessions` - the model's sessions; a call leases one while it runs
    /// * `builder` - the model's inputs, resolved at startup
    /// * `max_rows` - rows in one call, which every call is padded up to
    /// * `max_delay` - how long a call waits for more requests before running short.
    ///   Zero takes whatever is already queued and never waits
    /// * `lanes` - threads draining the queue, which is the calls that can be in flight
    pub(crate) fn start(
        sessions: Arc<Sessions>,
        builder: Arc<InputBuilder>,
        max_rows: usize,
        max_delay: Duration,
        lanes: usize,
    ) -> Self {
        let (queue, receiver) = mpsc::channel::<Pending>();
        let max_rows = max_rows.max(1);
        let waiting = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // One receiver shared by every lane, so a request is taken by whichever lane is
        // free. Holding the lock spans a `recv`, which is where a lane waits for work
        // anyway; the call itself runs with the lock released.
        let receiver = Arc::new(Mutex::new(receiver));

        for lane in 0..lanes.max(1) {
            let sessions = Arc::clone(&sessions);
            let builder = Arc::clone(&builder);
            let receiver = Arc::clone(&receiver);
            let waiting = Arc::clone(&waiting);
            std::thread::Builder::new()
                .name(format!("hushar-merge-{lane}"))
                .spawn(move || {
                    drain(
                        &sessions, &builder, max_rows, max_delay, &receiver, &waiting,
                    );
                })
                .expect("a merger thread is spawnable");
        }

        Self {
            queue,
            max_rows,
            waiting,
        }
    }

    /// Queues a request's rows, one entry per call they span.
    ///
    /// Returns one receiver per call, in row order, so the caller reassembles its scores
    /// by awaiting them in turn. A request of no rows queues nothing.
    pub(crate) fn submit(&self, columns: Arc<RequestColumns>) -> Vec<oneshot::Receiver<Reply>> {
        let total = columns.rows();
        let mut receivers = Vec::with_capacity(total.div_ceil(self.max_rows));
        let mut at = 0;
        while at < total {
            let end = (at + self.max_rows).min(total);
            let (reply, receiver) = oneshot::channel();
            let depth = self
                .waiting
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            // A closed queue means the merger's threads are gone. Dropping the sender
            // makes the caller's receiver fail rather than wait for a call that will
            // never run.
            if self
                .queue
                .send(Pending {
                    columns: Arc::clone(&columns),
                    rows: at..end,
                    reply,
                    queue_depth: depth,
                })
                .is_err()
            {
                self.waiting
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                break;
            }
            receivers.push(receiver);
            at = end;
        }
        receivers
    }
}

/// Takes requests off the queue and runs them as merged calls, until the queue closes.
fn drain(
    sessions: &Sessions,
    builder: &InputBuilder,
    max_rows: usize,
    max_delay: Duration,
    receiver: &Mutex<mpsc::Receiver<Pending>>,
    waiting: &std::sync::atomic::AtomicUsize,
) {
    // A request that did not fit the call being filled starts the next one.
    let mut carry: Option<Pending> = None;
    loop {
        let first = match carry.take() {
            Some(pending) => pending,
            None => {
                let guard = receiver.lock().unwrap_or_else(|e| e.into_inner());
                match guard.recv() {
                    Ok(pending) => {
                        waiting.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        pending
                    }
                    // Every sender is gone, so no further request can arrive.
                    Err(_) => return,
                }
            }
        };

        let mut rows = first.len();
        let mut batch = vec![first];
        let deadline = Instant::now() + max_delay;
        while rows < max_rows {
            let left = deadline.saturating_duration_since(Instant::now());
            let taken = {
                let guard = receiver.lock().unwrap_or_else(|e| e.into_inner());
                if left.is_zero() {
                    // No window: take what is already queued and run.
                    guard.try_recv().ok()
                } else {
                    guard.recv_timeout(left).ok()
                }
            };
            let Some(pending) = taken else { break };
            waiting.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            if rows + pending.len() > max_rows {
                carry = Some(pending);
                break;
            }
            rows += pending.len();
            batch.push(pending);
        }

        execute(sessions, builder, max_rows, batch);
    }
}

/// Builds one call from `batch`, runs it, and hands every request its own rows.
///
/// A failure fails exactly the requests in this call; the next call is unaffected,
/// because nothing about the failure is retained.
fn execute(sessions: &Sessions, builder: &InputBuilder, max_rows: usize, batch: Vec<Pending>) {
    let parts: Vec<(&RequestColumns, Range<usize>)> = batch
        .iter()
        .map(|pending| (pending.columns.as_ref(), pending.rows.clone()))
        .collect();
    let real_rows: usize = parts
        .iter()
        .map(|(_, rows)| rows.end.saturating_sub(rows.start))
        .sum();

    let build_start = Instant::now();
    let built = builder.build_many(&parts, max_rows);
    let built_at = Instant::now();
    drop(parts);

    let outcome = match built {
        Err(e) => Err(format!(
            "a merged call of {real_rows} row(s) could not be built: {e}"
        )),
        Ok(inputs) => {
            let session = sessions.acquire();
            let call_start = Instant::now();
            let produced = session.run(inputs);
            let call_end = Instant::now();
            drop(session);
            match produced {
                Err(e) => Err(format!("a merged call of {real_rows} row(s) failed: {e}")),
                Ok(produced) if produced.rows() != max_rows => Err(format!(
                    "a merged call of {max_rows} rows came back with {} rows",
                    produced.rows()
                )),
                Ok(produced) => Ok((produced, call_start, call_end)),
            }
        }
    };

    // Release every request's columns before replying, so the caller's own handle is the
    // last one left and it can take its features back for the inference log.
    let replies: Vec<(oneshot::Sender<Reply>, usize, usize)> = batch
        .into_iter()
        .map(|pending| {
            let rows = pending.rows.end.saturating_sub(pending.rows.start);
            (pending.reply, rows, pending.queue_depth)
        })
        .collect();

    let (produced, call_start, call_end) = match outcome {
        Err(message) => {
            for (reply, _, _) in replies {
                let _ = reply.send(Err(message.clone()));
            }
            return;
        }
        Ok(parts) => parts,
    };

    let Some(primary) = produced.first() else {
        for (reply, _, _) in replies {
            let _ = reply.send(Err("a merged call returned no outputs at all".to_owned()));
        }
        return;
    };
    let width = primary.width();
    let vec_time = (built_at - build_start).as_micros();
    let tensor_time = (call_start - built_at).as_micros();
    let inference_time = (call_end - call_start).as_micros();
    let ballast = max_rows.saturating_sub(real_rows) as u128;

    let mut offset = 0;
    for (reply, rows, queue_depth) in replies {
        let from = offset * width;
        let to = (offset + rows) * width;
        offset += rows;

        let taken = match &primary.data {
            ScoreData::F32(values) => values.get(from..to).map(|v| ScoreData::F32(v.to_vec())),
            ScoreData::F64(values) => values.get(from..to).map(|v| ScoreData::F64(v.to_vec())),
        };
        let slice = match taken {
            Some(scores) => {
                // Every request in the call shares its stage timings and its fill: the
                // call is one unit of work and no part of it belongs to one request.
                let mut report = RequestReport {
                    vec_time,
                    tensor_time,
                    inference_time,
                    engine_calls: 1,
                    ballast_rows: ballast,
                    queue_depth: queue_depth as u128,
                    ..RequestReport::default()
                };
                report.set_fill(real_rows, max_rows);
                Ok(ScoredBatch {
                    scores,
                    timings: report,
                })
            }
            None => Err(format!(
                "a merged call returned {} values, short of the rows it was given",
                primary.data.len()
            )),
        };
        let _ = reply.send(slice);
    }

    // A ladder of call widths would go here: rather than always padding to `max_rows`,
    // round the real rows up to the next of a few configured widths, so a light load
    // computes less ballast. Each width is another engine to compile on a provider that
    // compiles per shape, which is why one width is the default and the only one built.
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use crate::config::vectorization_config::VectorizationConfig;
    use crate::inference::backend::test_support::{
        FailingBackend, RecordingBackend, SlowBackend, SummingBackend,
    };
    use crate::inference::batch::{ElementType, IoSpec};
    use crate::inference::input_builder::test_support::{V, request, row};

    /// A builder taking one identity feature into a 1-wide input, defaulting to 7.
    pub(crate) fn builder() -> Arc<InputBuilder> {
        let config = VectorizationConfig::from_json(
            r#"{
                "feature_transformations": {"a": {"type": "identity", "default_val": [7.0]}},
                "feature_order": ["a"]
            }"#,
        )
        .expect("valid configuration");
        Arc::new(
            InputBuilder::resolve(
                config,
                &[IoSpec::new("features", ElementType::F32, Some(1))],
            )
            .expect("one feature into one input"),
        )
    }

    /// A pool holding `backend` alone, so one lane.
    pub(crate) fn pool_of(backend: Arc<dyn InferenceBackend>) -> Arc<Sessions> {
        Arc::new(SessionPool::new(vec![backend]))
    }

    /// `rows` rows whose single feature counts up from `base`.
    pub(crate) fn rows_from(base: usize, rows: usize) -> Arc<RequestColumns> {
        let ids: Vec<String> = (0..rows).map(|r| format!("r{}", base + r)).collect();
        let built: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(r, id)| row(id, &[("a", V::f32((base + r) as f32))]))
            .collect();
        Arc::new(request(&built))
    }

    /// Collects every reply for one submission, in row order.
    pub(crate) fn await_all(receivers: Vec<oneshot::Receiver<Reply>>) -> Vec<Reply> {
        receivers
            .into_iter()
            .map(|receiver| {
                receiver
                    .blocking_recv()
                    .unwrap_or_else(|_| Err("the merger dropped the reply".to_owned()))
            })
            .collect()
    }

    /// The scores of one reply as floats.
    pub(crate) fn floats(reply: &Reply) -> Vec<f32> {
        match reply {
            Ok(scored) => scored.scores.as_f32(),
            Err(e) => panic!("expected scores, got {e}"),
        }
    }

    /// Asserts two concurrent requests become exactly one engine call, and that each
    /// caller gets only its own rows back.
    #[test]
    fn two_requests_become_one_call_and_each_gets_its_own_rows() {
        let backend = Arc::new(RecordingBackend::new(1, 1));
        let merger = Merger::start(
            pool_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>),
            builder(),
            8,
            Duration::from_millis(200),
            1,
        );

        let first = rows_from(0, 3);
        let second = rows_from(10, 2);
        let first_replies = merger.submit(Arc::clone(&first));
        let second_replies = merger.submit(Arc::clone(&second));

        let first_out = await_all(first_replies);
        let second_out = await_all(second_replies);

        assert_eq!(
            backend.batches(),
            vec![8],
            "one call, padded to the configured width"
        );
        assert_eq!(first_out.len(), 1, "three rows fit one call");
        assert_eq!(second_out.len(), 1);
        assert_eq!(
            floats(&first_out[0]),
            vec![0.0, 1.0, 2.0],
            "the first caller gets its own three rows"
        );
        assert_eq!(
            floats(&second_out[0]),
            vec![10.0, 11.0],
            "and the second only its own two"
        );
    }

    /// Asserts a call runs short when the window expires rather than waiting for company
    /// that never arrives.
    #[test]
    fn the_window_expiring_still_serves_one_request() {
        let backend = Arc::new(RecordingBackend::new(1, 1));
        let merger = Merger::start(
            pool_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>),
            builder(),
            16,
            Duration::from_millis(30),
            1,
        );

        let only = rows_from(0, 2);
        let out = await_all(merger.submit(Arc::clone(&only)));

        assert_eq!(out.len(), 1);
        assert_eq!(floats(&out[0]), vec![0.0, 1.0]);
        assert_eq!(
            backend.batches(),
            vec![16],
            "the call still presents the one shape, mostly ballast"
        );
    }

    /// Asserts a request larger than the cap spans the fewest calls it can.
    #[test]
    fn a_request_above_the_cap_spans_the_fewest_calls() {
        let backend = Arc::new(RecordingBackend::new(1, 1));
        let merger = Merger::start(
            pool_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>),
            builder(),
            4,
            Duration::ZERO,
            1,
        );

        let big = rows_from(0, 9);
        let out = await_all(merger.submit(Arc::clone(&big)));

        assert_eq!(out.len(), 3, "nine rows at four per call is three calls");
        assert_eq!(
            backend.batches(),
            vec![4, 4, 4],
            "every call presents the same shape"
        );
        let all: Vec<f32> = out.iter().flat_map(floats).collect();
        assert_eq!(
            all,
            (0..9).map(|r| r as f32).collect::<Vec<f32>>(),
            "and the rows come back in request order across the calls"
        );
    }

    /// Asserts a failed call fails only its own requests, and the merger keeps serving.
    #[test]
    fn a_failed_call_fails_only_its_own_requests() {
        let pool = pool_of(Arc::new(FailingBackend));
        let merger = Merger::start(pool, builder(), 4, Duration::ZERO, 1);

        let columns = rows_from(0, 2);
        let out = await_all(merger.submit(Arc::clone(&columns)));
        assert_eq!(out.len(), 1);
        let Err(message) = &out[0] else {
            panic!("the engine always fails here");
        };
        assert!(
            message.contains("engine exploded"),
            "the engine's own words should reach the caller: {message}"
        );

        // The merger is still running: a second request is served, not ignored.
        let again = await_all(merger.submit(rows_from(5, 1)));
        assert_eq!(again.len(), 1, "the next call still happened");
        assert!(again[0].is_err(), "this backend fails every call");
    }

    /// Asserts a caller that goes away does not wedge the merger.
    ///
    /// The reply has nowhere to go, which must be ignored rather than treated as an
    /// error, or one abandoned request would stop every later one.
    #[test]
    fn a_dropped_caller_does_not_wedge_the_merger() {
        let backend = Arc::new(RecordingBackend::new(1, 1));
        let merger = Merger::start(
            pool_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>),
            builder(),
            4,
            Duration::ZERO,
            1,
        );

        let abandoned = rows_from(0, 2);
        drop(merger.submit(Arc::clone(&abandoned)));

        let served = await_all(merger.submit(rows_from(20, 2)));
        assert_eq!(served.len(), 1);
        assert_eq!(
            floats(&served[0]),
            vec![20.0, 21.0],
            "the request after the abandoned one is served normally"
        );
    }

    /// Asserts the caller gets its features back once its calls are done, which the
    /// inference log needs.
    #[test]
    fn the_caller_owns_its_columns_again_after_its_calls() {
        let backend = Arc::new(SummingBackend::new(1, 1));
        let merger = Merger::start(
            pool_of(backend as Arc<dyn InferenceBackend>),
            builder(),
            4,
            Duration::ZERO,
            1,
        );

        let columns = rows_from(0, 2);
        let out = await_all(merger.submit(Arc::clone(&columns)));
        assert!(out.iter().all(Result::is_ok));

        let owned = Arc::into_inner(columns)
            .expect("the merger released its handle before replying, so this is the last");
        let (row_ids, features) = owned.into_wire();
        assert_eq!(row_ids, vec!["r0".to_owned(), "r1".to_owned()]);
        assert!(features.contains_key("a"), "the features survived merging");
    }

    /// Asserts several lanes run several calls at once, so the sessions are all used.
    #[test]
    fn several_lanes_run_several_calls_at_once() {
        let sessions: Vec<Arc<dyn InferenceBackend>> = (0..2)
            .map(|_| {
                Arc::new(SlowBackend::new(1, 1, Duration::from_millis(80)))
                    as Arc<dyn InferenceBackend>
            })
            .collect();
        let pool = Arc::new(SessionPool::new(sessions));
        let merger = Merger::start(pool, builder(), 2, Duration::ZERO, 2);

        let started = Instant::now();
        let first = merger.submit(rows_from(0, 2));
        let second = merger.submit(rows_from(10, 2));
        let first_out = await_all(first);
        let second_out = await_all(second);
        let elapsed = started.elapsed();

        assert_eq!(floats(&first_out[0]), vec![0.0, 1.0]);
        assert_eq!(floats(&second_out[0]), vec![10.0, 11.0]);
        assert!(
            elapsed < Duration::from_millis(150),
            "two 80 ms calls on two lanes should overlap, took {elapsed:?}"
        );
    }

    /// Asserts an empty request queues nothing and so waits for nothing.
    #[test]
    fn an_empty_request_queues_no_call() {
        let backend = Arc::new(RecordingBackend::new(1, 1));
        let merger = Merger::start(
            pool_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>),
            builder(),
            4,
            Duration::ZERO,
            1,
        );

        let empty = Arc::new(request(&[]));
        assert!(
            merger.submit(empty).is_empty(),
            "no rows is no call to wait for"
        );
        assert!(
            backend.batches().is_empty(),
            "and the engine is never asked"
        );
    }
}

#[cfg(test)]
mod equivalence_tests {
    use super::tests_support::*;
    use super::*;
    use crate::inference::backend::test_support::SummingBackend;
    use crate::inference::execution_plan::ExecutionPlan;
    use crate::inference::scoring::{FeatureLogging, MiniBatch, score_features};

    /// Scores `columns` through the unmerged path under `plan`.
    fn scored_directly(
        backend: Arc<dyn InferenceBackend>,
        builder: &InputBuilder,
        columns: &RequestColumns,
        plan: &ExecutionPlan,
    ) -> Vec<f32> {
        score_features(
            backend.as_ref(),
            columns,
            builder,
            plan,
            FeatureLogging::Record,
        )
        .expect("scored")
        .scores
        .as_f32()
    }

    /// Asserts all three plans produce identical scores for the same request.
    ///
    /// The plan decides how many engine calls a request becomes. It must never decide
    /// what the request scores.
    #[test]
    fn every_plan_scores_a_request_identically() {
        let builder = builder();
        let columns = rows_from(0, 6);
        let expected: Vec<f32> = (0..6).map(|r| r as f32).collect();

        let whole = scored_directly(
            Arc::new(SummingBackend::new(1, 1)),
            &builder,
            &columns,
            &ExecutionPlan::WholeRequest,
        );
        assert_eq!(whole, expected, "whole request");

        let shared = scored_directly(
            Arc::new(SummingBackend::new(1, 1)),
            &builder,
            &columns,
            &ExecutionPlan::SharedSession {
                mini_batch: MiniBatch {
                    size: 2,
                    is_fixed: false,
                },
            },
        );
        assert_eq!(shared, expected, "mini batches on a shared session");

        let merger = Merger::start(
            pool_of(Arc::new(SummingBackend::new(1, 1))),
            Arc::clone(&builder),
            4,
            Duration::ZERO,
            1,
        );
        let merged: Vec<f32> = await_all(merger.submit(Arc::clone(&columns)))
            .iter()
            .flat_map(floats)
            .collect();
        assert_eq!(merged, expected, "merged across requests");

        assert_eq!(whole, merged, "every plan agrees, call shape aside");
    }

    /// Asserts many concurrent requests are each served their own rows exactly once,
    /// with nothing lost, duplicated, or attributed to the wrong caller.
    #[test]
    fn concurrent_requests_are_never_crossed() {
        const CALLERS: usize = 24;
        let merger = Arc::new(Merger::start(
            pool_of(Arc::new(SummingBackend::new(1, 1))),
            builder(),
            8,
            Duration::from_millis(5),
            2,
        ));

        let handles: Vec<_> = (0..CALLERS)
            .map(|caller| {
                let merger = Arc::clone(&merger);
                std::thread::spawn(move || {
                    // Each caller's rows are its own decade, so a crossed reply is
                    // visible rather than merely wrong by coincidence.
                    let base = caller * 100;
                    let columns = rows_from(base, 3);
                    let out: Vec<f32> = await_all(merger.submit(columns))
                        .iter()
                        .flat_map(floats)
                        .collect();
                    (caller, out)
                })
            })
            .collect();

        for handle in handles {
            let (caller, out) = handle.join().expect("no caller panics");
            let base = caller * 100;
            let expected: Vec<f32> = (0..3).map(|r| (base + r) as f32).collect();
            assert_eq!(
                out, expected,
                "caller {caller} must get exactly its own three rows"
            );
        }
    }

    /// Asserts a partly-filled call reports its ballast and its fill honestly.
    ///
    /// Four real rows in a call of ten is 40% full with six rows of ballast computed and
    /// thrown away, which is the cost of always presenting one shape.
    #[test]
    fn a_partly_filled_call_reports_its_ballast_and_fill() {
        let merger = Merger::start(
            pool_of(Arc::new(SummingBackend::new(1, 1))),
            builder(),
            10,
            Duration::ZERO,
            1,
        );

        let out = await_all(merger.submit(rows_from(0, 4)));
        let Ok(scored) = &out[0] else {
            panic!("the call succeeds");
        };
        assert_eq!(scored.timings.engine_calls, 1);
        assert_eq!(
            scored.timings.ballast_rows, 6,
            "ten rows sent, four real, six discarded"
        );
        assert_eq!(scored.timings.fill_percent, 40, "four of ten is 40% full");
        assert_eq!(
            scored.timings.lease_wait, 0,
            "a merged request holds no session of its own"
        );
        assert!(
            scored.timings.queue_depth >= 1,
            "it was at least the only entry in the queue"
        );
    }

    /// Asserts a full call reports no ballast, so the measure distinguishes a wasteful
    /// call from an efficient one.
    #[test]
    fn a_full_call_reports_no_ballast() {
        let merger = Merger::start(
            pool_of(Arc::new(SummingBackend::new(1, 1))),
            builder(),
            4,
            Duration::ZERO,
            1,
        );

        let out = await_all(merger.submit(rows_from(0, 4)));
        let Ok(scored) = &out[0] else {
            panic!("the call succeeds");
        };
        assert_eq!(scored.timings.ballast_rows, 0);
        assert_eq!(scored.timings.fill_percent, 100);
    }
}
