// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Leasing one reusable engine resource to one caller.
//!
//! The outer model pool holds interchangeable backend replicas. Neuron backends also use
//! the same primitive for mutable device tensor sets. A lease returns its value on drop,
//! including during unwinding, and waits rather than sharing when every value is busy.
//!
//! For ONNX Runtime, each replica owns one session and its engine thread pool. A request
//! takes an idle session for exactly as long as it scores, so two inferences never contend
//! on one session while another remains idle.
//!
//! The pool is per request rather than per thread on purpose. The earlier design gave each
//! blocking thread a session for life, indexed by a counter, on the assumption that the
//! blocking pool and the session count are one to one. Tokio blocking threads are not
//! permanent — one that idles for ten seconds exits and is replaced — and a replacement
//! took the *next* counter value, not the slot the dead thread vacated. After any quiet
//! spell two live threads could share one session while another stayed idle, and the
//! imbalance persisted into the next peak. A lease cannot drift like that: a session is
//! either held by exactly one request or idle.

use std::ops::{Deref, DerefMut};
use std::sync::{Condvar, Mutex};

/// A fixed set of sessions that requests borrow one at a time.
///
/// Holds the sessions not currently leased. Taking and returning one is a mutex-guarded
/// `Vec` pop or push, held for nanoseconds against an inference that runs for
/// milliseconds; the mutex also carries the condition variable a caller waits on when
/// every session is out.
#[derive(Debug)]
pub(crate) struct SessionPool<T> {
    idle: Mutex<Vec<T>>,
    returned: Condvar,
    total: usize,
}

/// One session, held until dropped.
///
/// Dereferences to the session. Dropping it — including during a panic's unwinding —
/// returns the session to the pool and wakes one waiter, so a session can never be lost.
#[derive(Debug)]
pub(crate) struct SessionLease<'a, T> {
    pool: &'a SessionPool<T>,
    session: Option<T>,
}

impl<T> SessionPool<T> {
    /// A pool holding every one of `sessions`.
    ///
    /// # Panics
    ///
    /// Panics when `sessions` is empty: a model with no session could serve nothing, and
    /// the configuration refuses that before a model is loaded.
    pub(crate) fn new(sessions: Vec<T>) -> Self {
        assert!(
            !sessions.is_empty(),
            "a session pool needs at least one session"
        );
        Self {
            total: sessions.len(),
            idle: Mutex::new(sessions),
            returned: Condvar::new(),
        }
    }

    /// Takes an idle session, waiting for one to be returned if every session is out.
    ///
    /// Waiting is the right response to exhaustion: the blocking pool is sized to the
    /// session count, so in the ordinary case no caller ever waits, and a caller that does
    /// — because the blocking pool was configured wider than the sessions — queues for a
    /// whole session rather than sharing one and halving two inferences at once.
    pub(crate) fn acquire(&self) -> SessionLease<'_, T> {
        let mut idle = self
            .idle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while idle.is_empty() {
            idle = self
                .returned
                .wait(idle)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        let session = idle.pop();
        SessionLease {
            pool: self,
            session,
        }
    }

    /// Sessions in the pool, leased or not.
    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Sessions currently idle.
    #[cfg(test)]
    pub(crate) fn idle(&self) -> usize {
        self.idle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

impl<T> Deref for SessionLease<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.session
            .as_ref()
            .expect("a lease holds its session until it is dropped")
    }
}

impl<T> DerefMut for SessionLease<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.session
            .as_mut()
            .expect("a lease holds its session until it is dropped")
    }
}

impl<T> Drop for SessionLease<'_, T> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool
                .idle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(session);
            self.pool.returned.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Asserts concurrent leases hold distinct sessions and every one comes back on drop.
    #[test]
    fn concurrent_leases_hold_distinct_sessions_and_return_them() {
        let pool = SessionPool::new(vec![0usize, 1, 2, 3]);
        assert_eq!(pool.total(), 4);
        assert_eq!(pool.idle(), 4);

        let leases: Vec<_> = (0..4).map(|_| pool.acquire()).collect();
        let mut held: Vec<usize> = leases.iter().map(|lease| **lease).collect();
        held.sort_unstable();
        assert_eq!(
            held,
            vec![0, 1, 2, 3],
            "four leases are the four sessions, once each"
        );
        assert_eq!(pool.idle(), 0);

        drop(leases);
        assert_eq!(
            pool.idle(),
            4,
            "every session is back once every lease is dropped"
        );
    }

    /// Asserts the failure the pool replaces cannot happen: with three sessions held, a
    /// brand-new thread gets the one idle session, never one that is busy.
    ///
    /// The per-thread counter this replaces would have handed a fresh thread the next index
    /// in sequence, which after any thread churn could be a busy session.
    #[test]
    fn a_fresh_thread_gets_the_idle_session_not_a_busy_one() {
        let pool = Arc::new(SessionPool::new(vec![10usize, 11, 12, 13]));
        // Churn: eight short-lived threads each take and return a session, as blocking
        // threads that idle out and are replaced would.
        for _ in 0..8 {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                let lease = pool.acquire();
                let _ = *lease;
            })
            .join()
            .unwrap();
        }
        let held: Vec<_> = (0..3).map(|_| pool.acquire()).collect();
        let busy: Vec<usize> = held.iter().map(|lease| **lease).collect();

        let got = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || *pool.acquire()).join().unwrap()
        };
        assert!(
            !busy.contains(&got),
            "the new thread got {got}, which was held by another request; busy were {busy:?}"
        );
        assert_eq!(
            pool.idle(),
            1,
            "the fresh thread's lease was returned when it exited"
        );
    }

    /// Asserts a caller who finds every session out waits for one instead of sharing.
    #[test]
    fn acquire_waits_for_a_busy_session_to_come_back() {
        let pool = Arc::new(SessionPool::new(vec!["only"]));
        let lease = pool.acquire();

        let waiter = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                let started = Instant::now();
                let lease = pool.acquire();
                (*lease, started.elapsed())
            })
        };
        thread::sleep(Duration::from_millis(80));
        assert_eq!(
            pool.idle(),
            0,
            "the waiter must not have conjured a second session"
        );
        drop(lease);

        let (session, waited) = waiter.join().unwrap();
        assert_eq!(session, "only");
        assert!(
            waited >= Duration::from_millis(40),
            "the waiter should have blocked until the lease was dropped, waited {waited:?}"
        );
        assert_eq!(pool.idle(), 1);
    }

    /// Asserts a session leased by a request that panics is returned all the same.
    #[test]
    fn a_panicking_holder_still_returns_its_session() {
        let pool = SessionPool::new(vec![1u8, 2]);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = pool.acquire();
            panic!("scoring failed");
        }));
        assert!(outcome.is_err(), "the panic propagates");
        assert_eq!(pool.idle(), 2, "the lease's drop ran during unwinding");
    }

    /// Asserts a pool with no sessions is refused at construction.
    #[test]
    #[should_panic(expected = "at least one session")]
    fn an_empty_pool_is_refused() {
        let _ = SessionPool::<u8>::new(Vec::new());
    }
}
