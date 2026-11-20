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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssymm(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _m: i64,
    _n: i64,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssymm_async(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _m: i64,
    _n: i64,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyrk(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyrk_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyr2k(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ssyr2k_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strmm(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _b: *mut f32,
    _ldb: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strmm_async(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _b: *mut f32,
    _ldb: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strsm(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _b: *mut f32,
    _ldb: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_strsm_async(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f32,
    _a: *const f32,
    _lda: i64,
    _b: *mut f32,
    _ldb: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsymm(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _m: i64,
    _n: i64,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsymm_async(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _m: i64,
    _n: i64,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyrk(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _n: i64,
    _k: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _beta: f64,
    _c: *mut f64,
    _ldc: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyrk_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
    _n: i64,
    _k: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _beta: f64,
    _c: *mut f64,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyr2k(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dsyr2k_async(
    _q: *mut c_void,
    _uplo: i32,
    _trans: i32,
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

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrmm(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _b: *mut f64,
    _ldb: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrmm_async(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _b: *mut f64,
    _ldb: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrsm(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _b: *mut f64,
    _ldb: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_dtrsm_async(
    _q: *mut c_void,
    _side: i32,
    _uplo: i32,
    _trans: i32,
    _diag: i32,
    _m: i64,
    _n: i64,
    _alpha: f64,
    _a: *const f64,
    _lda: i64,
    _b: *mut f64,
    _ldb: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}
