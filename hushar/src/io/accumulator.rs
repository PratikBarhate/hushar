// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Batching on the request path, with nothing behind it.
//!
//! The request path appends to a buffer here and, on the request that fills it, hands
//! the whole batch to a spawned task.
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
//! A mutex and a push, the same cost for every request. The request that fills the
//! buffer additionally takes the batch and spawns a task; it does **not** await the
//! send, so no request ever waits on S3, CloudWatch or Kinesis.
//!
//! The lock is [`std::sync::Mutex`] rather than tokio's, because the critical section
//! contains no `await` — it is a push and possibly a `mem::take`.
//!
//! # Why the sends are counted
//!
//! The sends in the air are bounded by a semaphore, and memory is bounded with them:
//!
//! ```text
//!   worst case held  =  capacity × (max_in_flight + 1)
//!                       └ buffer ┘   └ spawned sends ┘
//! ```
//!
//! When every permit is out, the batch is **dropped and reported** rather than
//! awaited: shedding observability keeps the service fast, whereas applying
//! backpressure would make a slow log destination into slow inference. The batch
//! dropped is the oldest, so what survives is the freshest data.

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
/// Dropping the permit is what frees a slot, so it has to outlive the send rather than
/// the call that started it.
#[derive(Debug)]
struct Batch<T> {
    items: Vec<T>,
    permit: tokio::sync::OwnedSemaphorePermit,
}

impl<T> Accumulator<T> {
    /// Builds an accumulator.
    ///
    /// # Arguments
    ///
    /// * `what` - names this accumulator in the line printed when a batch is shed
    /// * `capacity` - items per send. Zero is treated as one.
    /// * `max_in_flight` - sends allowed at once. Zero is treated as one, and the value
    ///   is capped at `u32::MAX`.
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

    /// Locks the buffer, recovering the guard even if a previous holder panicked.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<T>> {
        self.buffer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds `items`, spawning `send` with the whole batch once the buffer is full.
    ///
    /// Returns immediately in every case, and never awaits the send. The lock is
    /// dropped before the spawn, so a send that starts slowly does not hold up the
    /// requests filling the next batch. A full batch with no send slot free is dropped
    /// and reported rather than sent.
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
    /// Called at shutdown, before the runtime is dropped: the spawned sends are
    /// detached tasks, and a runtime shut down underneath them would drop their batches
    /// with no error anywhere. Acquiring every permit is what "no send is in the air"
    /// means, so the wait needs no polling and cannot race with a send that is just
    /// finishing.
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

    /// Counts sends and the items in them.
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

    /// A full buffer goes as one batch, and the buffer starts again from empty.
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

    /// Adding several items at once sends them in one batch, overshooting capacity.
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

    /// A flush sends a buffer that never reached capacity.
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

    /// Flushing an empty buffer sends nothing.
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

    /// A flush returns only once the spawned sends have landed.
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

    /// With every send slot taken, the next batch is shed and counted rather than held.
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

    /// A permit freed by a finished send becomes usable again.
    #[tokio::test]
    async fn a_slot_is_reusable_once_its_send_has_landed() {
        let accumulator: Accumulator<u32> = Accumulator::new("test", 1, 1);
        let sent = Arc::new(Sent::default());

        for i in 0..3 {
            let seen = Arc::clone(&sent);
            accumulator.add([i], move |items| async move { seen.record(items.len()) });
            accumulator.flush(|_| async {}).await;
        }

        assert_eq!(sent.sends(), 3, "every batch sent, none shed");
        assert_eq!(accumulator.shed(), 0);
    }

    /// A capacity and in-flight count of zero are clamped to one and still send.
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

    /// Concurrent adds lose no item and duplicate none.
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
