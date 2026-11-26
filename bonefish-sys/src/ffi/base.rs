// SPDX-License-Identifier: MIT OR Apache-2.0
//! Foreign declarations: queues, memory, events, devices, error text and test hooks. Split out of `lib.rs` (R15).

use std::ffi::{c_char, c_void};

unsafe extern "C" {
    /// `kind`: 0 default selector, 1 CPU, 2 GPU. Null on failure.
    pub fn bf_queue_new(kind: i32) -> *mut c_void;
    pub fn bf_queue_free(q: *mut c_void);
    /// Copies the NUL-terminated device name into `buf` (truncated to `cap`); returns the
    /// full length, or -1 on failure.
    pub fn bf_queue_device_name(q: *mut c_void, buf: *mut c_char, cap: i64) -> i64;
    /// `kind`: 0 shared, 1 device, 2 host USM. Null on failure.
    pub fn bf_malloc(q: *mut c_void, kind: i32, bytes: i64) -> *mut c_void;
    pub fn bf_free(q: *mut c_void, p: *mut c_void);
    pub fn bf_memcpy(q: *mut c_void, dst: *mut c_void, src: *const c_void, bytes: i64) -> i32;
    /// Number of SYCL devices, or -1 on failure.
    pub fn bf_device_count() -> i64;
    /// One piece of information about device `index`, copied NUL-terminated into `buf`
    /// (truncated to `cap`). Returns the full length, -1 on failure, or -2 when the device
    /// does not offer it. `field`: 0 name, 1 vendor, 2 backend, 3 device type, 4 driver
    /// version, 5 PCI address, 6 UUID (32 hex digits), 7 Intel device id (decimal).
    pub fn bf_device_info(index: i64, field: i32, buf: *mut c_char, cap: i64) -> i64;
    /// A queue with options: `device_index` >= 0 picks that device of the enumeration, -1
    /// picks by `kind` (0 default, 1 CPU, 2 GPU). `flags` bit 0 enables profiling. Always
    /// in-order, with the async handler. Null on failure.
    pub fn bf_queue_new_ex(kind: i32, device_index: i64, flags: i32) -> *mut c_void;
    /// Device start/end of the work behind an event, in nanoseconds on the device clock.
    /// Waits for the event. Needs a queue made with profiling (otherwise status 1).
    pub fn bf_event_profiling(e: *mut c_void, start_ns: *mut u64, end_ns: *mut u64) -> i32;
    /// An empty kernel, chained like the other asynchronous calls: the launch floor. Null on
    /// failure.
    pub fn bf_noop_async(q: *mut c_void, deps: *const *mut c_void, n_deps: i64) -> *mut c_void;
    /// A queue (in-order, with the async handler) on device `index`. Null on failure.
    pub fn bf_queue_new_device(index: i64) -> *mut c_void;
    /// 1 if the queue is in-order (every queue from `bf_queue_new` is), 0 if not, -1 on failure.
    pub fn bf_queue_is_in_order(q: *mut c_void) -> i32;
    /// Waits for all work on the queue. Returns 1 (message in `bf_last_error`) if that surfaced
    /// an asynchronous error.
    pub fn bf_queue_wait(q: *mut c_void) -> i32;
    /// Waits for one event (`wait_and_throw`); same status convention as `bf_queue_wait`.
    pub fn bf_event_wait(e: *mut c_void) -> i32;
    pub fn bf_event_free(e: *mut c_void);
    /// Diagnostic: submits a host task that throws; the failure is asynchronous. Returns the
    /// event, or null on failure.
    pub fn bf_test_async_failure(q: *mut c_void) -> *mut c_void;
    /// Diagnostic: `1` chains every asynchronous submission to the queue's previous one (the
    /// default), `0` turns that off. For A/B runs and for retesting a newer oneMKL.
    pub fn bf_test_set_async_chain(on: i32);
    /// Diagnostic: how many oneMKL calls have entered the shim's lock path. A test uses it to
    /// prove no oneMKL-calling function bypasses the lock. `-1` in a build without the shim.
    pub fn bf_mkl_lock_acquisitions() -> i64;
    /// Message of the last failure on this thread.
    pub fn bf_last_error() -> *const c_char;
}
