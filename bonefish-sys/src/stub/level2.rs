// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stand-ins for: level 2 BLAS (build without the shim). Split out of `lib.rs` (R15).

use super::base::STATUS_UNAVAILABLE;
use std::ffi::c_void;

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_sger(
    _q: *mut c_void,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _a: *mut f32,
    _lda: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssymv(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _x: *const f32,
    _incx: i64,
    _beta: f32,
    _y: *mut f32,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyr(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _a: *mut f32,
    _lda: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyr2(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _a: *mut f32,
    _lda: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strmv(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f32,
    _lda: i64,
    _x: *mut f32,
    _incx: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strsv(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f32,
    _lda: i64,
    _x: *mut f32,
    _incx: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_sgbmv(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _kl: i64,
    _ku: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _x: *const f32,
    _incx: i64,
    _beta: f32,
    _y: *mut f32,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsymv(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _x: *const f64,
    _incx: i64,
    _beta: f64,
    _y: *mut f64,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyr(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _a: *mut f64,
    _lda: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyr2(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _a: *mut f64,
    _lda: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrmv(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f64,
    _lda: i64,
    _x: *mut f64,
    _incx: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrsv(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f64,
    _lda: i64,
    _x: *mut f64,
    _incx: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dgbmv(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _kl: i64,
    _ku: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _x: *const f64,
    _incx: i64,
    _beta: f64,
    _y: *mut f64,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_sgemv(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _x: *const f32,
    _incx: i64,
    _beta: f32,
    _y: *mut f32,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dger(
    _q: *mut c_void,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _a: *mut f64,
    _lda: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dgemv(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _x: *const f64,
    _incx: i64,
    _beta: f64,
    _y: *mut f64,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_sger_async(
    _q: *mut c_void,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _a: *mut f32,
    _lda: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssymv_async(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _x: *const f32,
    _incx: i64,
    _beta: f32,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyr_async(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _a: *mut f32,
    _lda: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyr2_async(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _a: *mut f32,
    _lda: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strmv_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f32,
    _lda: i64,
    _x: *mut f32,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strsv_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f32,
    _lda: i64,
    _x: *mut f32,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_sgbmv_async(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _kl: i64,
    _ku: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _x: *const f32,
    _incx: i64,
    _beta: f32,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsymv_async(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _x: *const f64,
    _incx: i64,
    _beta: f64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyr_async(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _a: *mut f64,
    _lda: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyr2_async(
    _q: *mut c_void,
    _uplo: i32,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _a: *mut f64,
    _lda: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrmv_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f64,
    _lda: i64,
    _x: *mut f64,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrsv_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _n: i64,
    _a: *const f64,
    _lda: i64,
    _x: *mut f64,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dgbmv_async(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _kl: i64,
    _ku: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _x: *const f64,
    _incx: i64,
    _beta: f64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_sgemv_async(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _x: *const f32,
    _incx: i64,
    _beta: f32,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dger_async(
    _q: *mut c_void,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _a: *mut f64,
    _lda: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dgemv_async(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _x: *const f64,
    _incx: i64,
    _beta: f64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_cgemv_async(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _alpha: *const f32,
    _a: *const f32,
    _lda: i64,
    _x: *const f32,
    _incx: i64,
    _beta: *const f32,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zgemv_async(
    _q: *mut c_void,
    _trans: i32,
    _m: i64,
    _n: i64,
    _alpha: *const f64,
    _a: *const f64,
    _lda: i64,
    _x: *const f64,
    _incx: i64,
    _beta: *const f64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}
