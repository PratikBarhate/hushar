// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Batching on the request path, with nothing behind it.
//!
//! This replaces the sidecars. A sink used to hand its work to a worker on another
//! runtime through a bounded channel; now the request path appends to a buffer here
//! and, on the request that fills it, hands the whole batch to a spawned task. There
//! is no channel to fill, no worker to schedule, and no thread reserved away from
//! serving.
//!
//! ```text
//!   request  ──► add() ──► [ buffer ]  ── full ──►  tokio::spawn(send)
//!                              │                          │
//!                          lock, push                 permit held
//!                          (microseconds)             until it lands
//! ```
//!
//! # What each request pays
//!
//! A mutex, a push, and nothing else — the same cost for every request, which is the
//! one property worth preserving from the sidecar. The request that fills the buffer
//! additionally takes the batch and spawns a task; it does **not** await the send, so
//! no request ever waits on S3, CloudWatch or Kinesis. That is the whole reason to
//! spawn rather than `await` inline: awaiting would put one object-store round trip
//! onto one request in every `capacity`, which is a tail nobody can explain from the
//! outside.
//!
//! The lock is [`std::sync::Mutex`] rather than tokio's, because the critical section
//! contains no `await` — it is a push and possibly a `mem::take`. An async mutex here
//! would cost a state machine to protect a memcpy.
//!
//! # Why the sends are counted
//!
//! Removing the channel removed a bound, and an unbounded number of spawned sends
//! against a slow destination is an out-of-memory rather than a lost metric. So the
//! sends in the air are bounded by a semaphore, and memory is bounded with them:
//!
//! ```text
//!   worst case held  =  capacity × (max_in_flight + 1)
//!                       └ buffer ┘   └ spawned sends ┘
//! ```
//!
//! When every permit is out, the batch is **dropped and reported** rather than
//! awaited. That is the same trade the sidecar's full channel made and it is the
//! right one on a serving path: shedding observability keeps the service fast,
//! whereas applying backpressure would make a slow log destination into slow
//! inference. The batch dropped is the oldest, so what survives is the freshest data,
//! which is what an operator looking at a live problem wants.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Semaphore;

/// A buffer that hands full batches to a spawned send.
///
/// `T` is whatever the destination's API takes one of: a CloudWatch data point, an
/// inference log batch.
#[derive(Debug)]
pub(crate) struct Accumulator<T> {
    /// Names this accumulator in the line printed when a batch is shed.
    what: &'static str,
    capacity: usize,
    /// `u32` because that is what draining every permit takes; a wider count could
    /// not be drained, so a flush would return while a send was still in the air.
    max_in_flight: u32,
    buffer: std::sync::Mutex<Vec<T>>,
    /// One permit per send allowed to be in the air.
    sends: Arc<Semaphore>,
    shed: AtomicU64,
}

/// A full batch, and the permit that keeps its send counted.
///
/// The permit lives in here so it cannot be forgotten: dropping it is what frees a
/// slot, so it has to outlive the send rather than the call that started it.
#[derive(Debug)]
struct Batch<T> {
    items: Vec<T>,
    permit: tokio::sync::OwnedSemaphorePermit,
}

impl<T> Accumulator<T> {
    /// `capacity` items per send, and `max_in_flight` sends at once. Zero of either
    /// is treated as one, so a misconfiguration degrades to sending eagerly rather
    /// than buffering for ever or spawning nothing.
    pub(crate) fn new(what: &'static str, capacity: usize, max_in_flight: usize) -> Self {
        let capacity = capacity.max(1);
        let max_in_flight = max_in_flight.clamp(1, u32::MAX as usize) as u32;
        Self {
            what,
            capacity,
            max_in_flight,
            buffer: std::sync::Mutex::new(Vec::with_capacity(capacity)),
            sends: Arc::new(Semaphore::new(max_in_flight as usize)),
            shed: AtomicU64::new(0),
        }
    }

    /// Items per send, after clamping. Printed at startup, because a value that was
    /// clamped by an API limit is otherwise invisible.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Recovers the guard even if a previous holder panicked.
    ///
    /// Observability is not worth propagating a poisoning panic into the service.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<T>> {
        self.buffer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds `items`, spawning `send` with the whole batch once the buffer is full.
    ///
    /// Returns immediately in every case. The lock is dropped before the spawn, so a
    /// send that starts slowly does not hold up the requests filling the next batch.
    ///
    /// # Panics
    ///
    /// If called outside a tokio runtime, which on a serving path means it was called
    /// from somewhere other than a request handler.
    pub(crate) fn add<I, F, Fut>(&self, items: I, send: F)
    where
        I: IntoIterator<Item = T>,
        T: Send + 'static,
        F: FnOnce(Vec<T>) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let full = {
            let mut buffer = self.lock();
            buffer.extend(items);
            // Replaced rather than `mem::take`n, which would leave a zero-capacity
            // vector to grow again by doubling over the next `capacity` pushes. One
            // allocation per batch, in the branch that already runs once per batch.
            (buffer.len() >= self.capacity)
                .then(|| std::mem::replace(&mut *buffer, Vec::with_capacity(self.capacity)))
        };
        let Some(items) = full else { return };

        if let Some(batch) = self.reserve(items) {
            tokio::spawn(async move {
                let _permit = batch.permit;
                send(batch.items).await;
            });
        }
    }

    /// Claims a slot for `items`, or reports them lost if every send is still in the air.
    fn reserve(&self, items: Vec<T>) -> Option<Batch<T>> {
        match Arc::clone(&self.sends).try_acquire_owned() {
            Ok(permit) => Some(Batch { items, permit }),
            Err(_) => {
                let dropped = items.len() as u64;
                let total = self.shed.fetch_add(dropped, Ordering::Relaxed) + dropped;
                eprintln!(
                    "{}: {} sends still in flight, so {dropped} buffered records were \
                     dropped to keep the newest ({total} dropped since start); the \
                     destination is not keeping up",
                    self.what, self.max_in_flight
                );
                None
            }
        }
    }

    /// Sends whatever is buffered and waits for every send already in the air.
    ///
    /// Called at shutdown, which is exactly the window where the data matters most:
    /// a partial buffer thrown away is the record of what the service was doing when
    /// it stopped. Waiting for the spawned sends matters just as much — they are
    /// detached tasks, and a runtime shut down underneath them would drop their
    /// batches with no error anywhere.
    ///
    /// Acquiring every permit is what "no send is in the air" means, so the wait
    /// needs no polling and cannot race with a send that is just finishing.
    ///
    /// Assumes recording has stopped. A [`Self::add`] arriving while this holds the
    /// permits would find none free and shed, so the caller flushes only after the
    /// listener has stopped and in-flight requests have finished — which is what
    /// `serve_with_shutdown` guarantees before it returns.
    pub(crate) async fn flush<F, Fut>(&self, send: F)
    where
        F: FnOnce(Vec<T>) -> Fut,
        Fut: Future<Output = ()>,
    {
        let items = std::mem::take(&mut *self.lock());
        if !items.is_empty() {
            send(items).await;
        }
        let _ = self.sends.acquire_many(self.max_in_flight).await;
    }

    #[cfg(test)]
    pub(crate) fn buffered(&self) -> usize {
        self.lock().len()
    }

    #[cfg(test)]
    pub(crate) fn shed(&self) -> u64 {
        self.shed.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.max_in_flight as usize - self.sends.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// Counts sends and the items in them, so a test can assert on what was handed on
    /// rather than on where it went.
    #[derive(Debug, Default)]
    struct Sent {
        sends: AtomicUsize,
        items: AtomicUsize,
    }

    impl Sent {
        fn record(&self, n: usize) {
            self.sends.fetch_add(1, Ordering::SeqCst);
            self.items.fetch_add(n, Ordering::SeqCst);
        }
        fn sends(&self) -> usize {
            self.sends.load(Ordering::SeqCst)
        }
        fn items(&self) -> usize {
            self.items.load(Ordering::SeqCst)
        }
    }

    #[tokio::test]
    async fn nothing_is_sent_before_the_buffer_is_full() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 3, 2);
        let sent = Arc::new(Sent::default());

        for i in 0..2 {
            let seen = Arc::clone(&sent);
            accumulator.add([i], move |items| async move { seen.record(items.len()) });
        }

        assert_eq!(accumulator.buffered(), 2, "two of three");
        assert_eq!(sent.sends(), 0, "and nothing sent yet");
    }

    /// The whole batch goes at once, and the buffer starts again from empty -- so an
    /// item cannot be sent twice.
    #[tokio::test]
    async fn reaching_capacity_sends_the_whole_batch_exactly_once() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 3, 2);
        let sent = Arc::new(Sent::default());

        for i in 0..7 {
            let seen = Arc::clone(&sent);
            accumulator.add([i], move |items| async move { seen.record(items.len()) });
        }
        let seen = Arc::clone(&sent);
        accumulator
            .flush(move |items| async move { seen.record(items.len()) })
            .await;

        assert_eq!(sent.sends(), 3, "two full batches, then the flush");
        assert_eq!(sent.items(), 7, "every item sent, none twice");
        assert_eq!(accumulator.buffered(), 0);
    }

    /// CloudWatch adds three data points per scored batch, so adding several at once
    /// has to be able to cross the capacity rather than land exactly on it.
    #[tokio::test]
    async fn adding_several_at_once_can_overshoot_capacity() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 4, 2);
        let sent = Arc::new(Sent::default());
        let seen = Arc::clone(&sent);

        accumulator.add([1, 2, 3, 4, 5], move |items| async move {
            seen.record(items.len())
        });
        accumulator.flush(|_| async {}).await;

        assert_eq!(sent.sends(), 1, "one send, not one per item");
        assert_eq!(sent.items(), 5, "carrying all five, not truncated to four");
    }

    /// Whatever is buffered at shutdown is the record of what the service was doing
    /// when it stopped.
    #[tokio::test]
    async fn flushing_sends_a_partial_buffer() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 100, 2);
        let sent = Arc::new(Sent::default());
        let seen = Arc::clone(&sent);
        accumulator.add([1], move |items| async move { seen.record(items.len()) });

        let seen = Arc::clone(&sent);
        accumulator
            .flush(move |items| async move { seen.record(items.len()) })
            .await;

        assert_eq!(sent.items(), 1);
        assert_eq!(accumulator.buffered(), 0);
    }

    /// A service can stop before a single request arrives, and an empty send would be
    /// a wasted API call or an empty object in the prefix.
    #[tokio::test]
    async fn flushing_nothing_sends_nothing() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 10, 2);
        let sent = Arc::new(Sent::default());
        let seen = Arc::clone(&sent);

        accumulator
            .flush(move |items| async move { seen.record(items.len()) })
            .await;
        assert_eq!(sent.sends(), 0);
    }

    /// The property that makes shutdown correct: `flush` returns only once the
    /// spawned sends have landed. Without waiting on the permits this returns while
    /// the send is still sleeping, and a runtime shutting down would drop it.
    #[tokio::test]
    async fn flushing_waits_for_sends_already_in_flight() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 1, 2);
        let sent = Arc::new(Sent::default());
        let seen = Arc::clone(&sent);

        accumulator.add([1], move |items| async move {
            tokio::time::sleep(Duration::from_millis(80)).await;
            seen.record(items.len());
        });

        accumulator.flush(|_| async {}).await;
        assert_eq!(
            sent.sends(),
            1,
            "the flush returned before the spawned send finished"
        );
        assert_eq!(accumulator.in_flight(), 0);
    }

    /// Removing the channel removed a bound, so this is the bound that replaced it: a
    /// destination that stops draining costs a reported loss, not unbounded memory.
    ///
    /// One permit and a send that never finishes, so the second batch has nowhere to go.
    #[tokio::test]
    async fn a_saturated_destination_sheds_rather_than_growing_without_limit() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 2, 1);
        let sent = Arc::new(Sent::default());

        let seen = Arc::clone(&sent);
        accumulator.add([1, 2], move |items| async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            seen.record(items.len());
        });
        tokio::task::yield_now().await;
        assert_eq!(accumulator.in_flight(), 1, "the only permit is taken");

        let seen = Arc::clone(&sent);
        accumulator.add([3, 4], move |items| async move { seen.record(items.len()) });

        assert_eq!(accumulator.shed(), 2, "the second batch was dropped");
        assert_eq!(
            accumulator.buffered(),
            0,
            "and the buffer starts again, so the newest records are the ones kept"
        );
    }

    /// A permit freed by a send that finished has to become usable again, or the first
    /// slow moment would shed for ever.
    #[tokio::test]
    async fn a_slot_is_reusable_once_its_send_has_landed() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 1, 1);
        let sent = Arc::new(Sent::default());

        for i in 0..3 {
            let seen = Arc::clone(&sent);
            accumulator.add([i], move |items| async move { seen.record(items.len()) });
            // Let the spawned send run to completion, releasing its permit.
            accumulator.flush(|_| async {}).await;
        }

        assert_eq!(sent.sends(), 3, "every batch sent, none shed");
        assert_eq!(accumulator.shed(), 0);
    }

    /// Zero would either buffer for ever or spawn nothing, and both are worse than
    /// sending eagerly.
    #[tokio::test]
    async fn degenerate_settings_are_clamped_rather_than_disabling_the_sink() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 0, 0);
        assert_eq!(accumulator.capacity(), 1);

        let sent = Arc::new(Sent::default());
        let seen = Arc::clone(&sent);
        accumulator.add([1], move |items| async move { seen.record(items.len()) });
        accumulator.flush(|_| async {}).await;
        assert_eq!(sent.items(), 1, "a capacity of zero still sends");
    }

    /// The request path is shared, so the buffer has to be too: many tasks adding at
    /// once must lose nothing and double-count nothing.
    #[tokio::test]
    async fn concurrent_adds_neither_lose_nor_duplicate_an_item() {
        let accumulator: Arc<Accumulator<u32>> = Arc::new(Accumulator::new("test", 10, 8));
        let sent = Arc::new(Sent::default());

        let mut handles = Vec::new();
        for i in 0..64u32 {
            let accumulator = Arc::clone(&accumulator);
            let seen = Arc::clone(&sent);
            handles.push(tokio::spawn(async move {
                accumulator.add([i], move |items| async move { seen.record(items.len()) });
            }));
        }
        for handle in handles {
            handle.await.expect("adder");
        }

        let seen = Arc::clone(&sent);
        accumulator
            .flush(move |items| async move { seen.record(items.len()) })
            .await;

        assert_eq!(sent.items(), 64);
        assert_eq!(accumulator.shed(), 0, "eight slots is ample for six sends");
    }
}
