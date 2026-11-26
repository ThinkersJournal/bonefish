// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stand-ins for: complex and real DFT (build without the shim). Split out of `lib.rs` (R15).

use super::base::STATUS_UNAVAILABLE;
use std::ffi::c_void;

pub unsafe extern "C" fn bf_dft_c2c_new(
    _q: *mut c_void,
    _prec: i32,
    _ndim: i64,
    _lengths: *const i64,
    _inplace: i32,
    _fwd_scale: f64,
    _bwd_scale: f64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dft_free(_h: *mut c_void) {}

pub unsafe extern "C" fn bf_dft_compute(
    _h: *mut c_void,
    _dir: i32,
    _input: *mut c_void,
    _out: *mut c_void,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dft_compute_async(
    _h: *mut c_void,
    _dir: i32,
    _input: *mut c_void,
    _out: *mut c_void,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dftr_new(
    _q: *mut c_void,
    _prec: i32,
    _ndim: i64,
    _lengths: *const i64,
    _fwd_scale: f64,
    _bwd_scale: f64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dftr_free(_h: *mut c_void) {}

pub unsafe extern "C" fn bf_dftr_compute(
    _h: *mut c_void,
    _dir: i32,
    _input: *mut c_void,
    _out: *mut c_void,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dftr_compute_async(
    _h: *mut c_void,
    _dir: i32,
    _input: *mut c_void,
    _out: *mut c_void,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}
