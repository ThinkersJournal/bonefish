//! Stand-ins for: level 3 BLAS (`gemm`) (build without the shim). Split out of `lib.rs` (R15).

use super::base::STATUS_UNAVAILABLE;
use std::ffi::c_void;

pub unsafe extern "C" fn bf_sgemm(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _b: *const f32,
    _ldb: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dgemm(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _b: *const f64,
    _ldb: i64,
    _beta: f64,
    _c: *mut f64,
    _ldc: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_sgemm_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _b: *const f32,
    _ldb: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dgemm_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _b: *const f64,
    _ldb: i64,
    _beta: f64,
    _c: *mut f64,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_cgemm(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: *const f32,
    _a: *const f32,
    _lda: i64,
    _b: *const f32,
    _ldb: i64,
    _beta: *const f32,
    _c: *mut f32,
    _ldc: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_zgemm(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: *const f64,
    _a: *const f64,
    _lda: i64,
    _b: *const f64,
    _ldb: i64,
    _beta: *const f64,
    _c: *mut f64,
    _ldc: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_cgemm_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: *const f32,
    _a: *const f32,
    _lda: i64,
    _b: *const f32,
    _ldb: i64,
    _beta: *const f32,
    _c: *mut f32,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_zgemm_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: *const f64,
    _a: *const f64,
    _lda: i64,
    _b: *const f64,
    _ldb: i64,
    _beta: *const f64,
    _c: *mut f64,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}
