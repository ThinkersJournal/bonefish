//! A queue freed right after a wait must not take the process down when the runtime posts the
//! task's error a moment late (ROADMAP R10).
//!
//! The scenario: threads each loop { new queue, a host task that fails, free its event, ONE wait,
//! free the queue }. With several threads running, the wait sometimes returns before the runtime
//! has posted the error; if the queue is already destroyed when it arrives, SYCL's default async
//! handler prints "Default async_handler caught exceptions" and calls `terminate`, killing the
//! whole test process. The shim therefore retires queues and destroys them after a grace period.
//!
//! This file is separate from `async_shim.rs` on purpose: when the bug is present this process
//! aborts rather than reporting a failure, and it must not take other tests with it.
//!
//! Measured on the Windows laptop (oneAPI 2026.1, CPU device) with the queue destroyed at once:
//! `four_threads` died in 25 of 25 processes, `one_thread` in 0 of 25. To repeat that A/B without
//! rebuilding: `BONEFISH_UNSAFE_QUEUE_GRACE_MS=0`. Linux: the same abort was seen in CI
//! (async_shim, 5 of 180 runs on main); this exact test has not been run in a loop there.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;

fn run(threads: usize, iterations: usize) {
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            std::thread::spawn(move || {
                for _ in 0..iterations {
                    // SAFETY: plain calls on a queue this thread owns; each handle is freed once.
                    unsafe {
                        let q = bf_queue_new(1);
                        assert!(!q.is_null(), "no CPU SYCL device");
                        let e = bf_test_async_failure(q);
                        assert!(!e.is_null());
                        bf_event_free(e);
                        // One wait only, whatever it returns: a late error is the point.
                        let _ = bf_queue_wait(q);
                        bf_queue_free(q);
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

/// The control: one thread never hit the problem, with or without the fix. If this fails, the
/// harness itself is broken (no device, no failing task), not the queue lifetime.
#[test]
fn one_thread_drops_queues_after_a_wait() {
    run(1, 200);
}

/// The case that killed the process: several threads, each dropping its queue right after one wait.
#[test]
fn four_threads_drop_queues_right_after_a_wait() {
    run(4, 50);
}
