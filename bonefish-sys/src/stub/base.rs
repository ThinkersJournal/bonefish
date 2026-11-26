// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stand-ins for: queues, memory, events, devices, error text and test hooks (build without the shim). Split out of `lib.rs` (R15).

use std::ffi::{c_char, c_void};

/// Returned by every status-returning call in a build without the shim.
pub const STATUS_UNAVAILABLE: i32 = 4;
pub(super) const UNAVAILABLE_MSG: &[u8] =
    b"bonefish was built without oneAPI (icpx not found at build time)\0";

pub unsafe extern "C" fn bf_queue_new(_kind: i32) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_queue_free(_q: *mut c_void) {}

pub unsafe extern "C" fn bf_queue_device_name(
    _q: *mut c_void,
    _buf: *mut c_char,
    _cap: i64,
) -> i64 {
    -1
}

pub unsafe extern "C" fn bf_malloc(_q: *mut c_void, _kind: i32, _bytes: i64) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_free(_q: *mut c_void, _p: *mut c_void) {}

pub unsafe extern "C" fn bf_memcpy(
    _q: *mut c_void,
    _dst: *mut c_void,
    _src: *const c_void,
    _bytes: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_device_count() -> i64 {
    -1
}

pub unsafe extern "C" fn bf_device_info(
    _index: i64,
    _field: i32,
    _buf: *mut c_char,
    _cap: i64,
) -> i64 {
    -1
}

pub unsafe extern "C" fn bf_queue_new_ex(
    _kind: i32,
    _device_index: i64,
    _flags: i32,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_event_profiling(
    _e: *mut c_void,
    _start_ns: *mut u64,
    _end_ns: *mut u64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_noop_async(
    _q: *mut c_void,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_queue_new_device(_index: i64) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_queue_is_in_order(_q: *mut c_void) -> i32 {
    -1
}

pub unsafe extern "C" fn bf_queue_wait(_q: *mut c_void) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_event_wait(_e: *mut c_void) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_event_free(_e: *mut c_void) {}

pub unsafe extern "C" fn bf_test_async_failure(_q: *mut c_void) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_test_set_async_chain(_on: i32) {}

pub unsafe extern "C" fn bf_mkl_lock_acquisitions() -> i64 {
    -1
}

pub unsafe extern "C" fn bf_last_error() -> *const c_char {
    UNAVAILABLE_MSG.as_ptr().cast()
}
