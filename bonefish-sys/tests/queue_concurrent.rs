// SPDX-License-Identifier: MIT OR Apache-2.0
//! Several threads creating the first queues of a process at once must all get one (bonefish#40).
//!
//! On Windows the first use of the SYCL runtime loads Intel's OpenCL runtime, and `tbb12`'s DllMain
//! calls `WinVerifyTrust` on itself while holding the loader lock; an NVIDIA driver thread does the
//! same from another thread, and the two deadlocked inside CRYPT32's default chain engine (seen in
//! `tests/level3.rs` with its eight tests starting together, 4 of 4 processes within 35 runs). The
//! shim now creates that engine once before the runtime loads (`warm_trust_engine`, common.hpp).
//! This test is the shape that hung: every thread's first queue, released together. A recurrence
//! fails by name through the watchdog instead of hanging the run; the deadlock is a race, so a
//! pass here is evidence only, the 4-process loop in the issue is the real measurement.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
mod common;

#[test]
fn many_threads_make_their_first_queue_together() {
    common::guarded(|| {
        common::arm();
        const THREADS: usize = 8;
        let start = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let start = start.clone();
                std::thread::spawn(move || {
                    start.wait();
                    // SAFETY: a queue this thread owns, freed once.
                    unsafe {
                        let q = bf_queue_new(1);
                        assert!(!q.is_null(), "no CPU SYCL device");
                        bf_queue_free(q);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("a queue-making thread panicked");
        }
    });
}
