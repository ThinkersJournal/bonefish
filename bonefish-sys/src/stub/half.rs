//! Stand-ins for: half/bf16 `gemm` and strided batched half/bf16 `gemm` (build without the shim). Split out
//! of `level3.rs` (R15, MOD-005): the same signatures as `ffi/level3.rs`, each returns a null event.

use std::ffi::c_void;

pub unsafe extern "C" fn bf_hgemm_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _b: *const u16,
    _ldb: i64,
    _beta: f32,
    _c: *mut u16,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_hgemm_f32_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _b: *const u16,
    _ldb: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_bgemm_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _b: *const u16,
    _ldb: i64,
    _beta: f32,
    _c: *mut u16,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_bgemm_f32_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _b: *const u16,
    _ldb: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_hgemm_batch_strided_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _stride_a: i64,
    _b: *const u16,
    _ldb: i64,
    _stride_b: i64,
    _beta: f32,
    _c: *mut u16,
    _ldc: i64,
    _stride_c: i64,
    _batch: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_hgemm_f32_batch_strided_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _stride_a: i64,
    _b: *const u16,
    _ldb: i64,
    _stride_b: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
    _stride_c: i64,
    _batch: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_bgemm_batch_strided_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _stride_a: i64,
    _b: *const u16,
    _ldb: i64,
    _stride_b: i64,
    _beta: f32,
    _c: *mut u16,
    _ldc: i64,
    _stride_c: i64,
    _batch: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_bgemm_f32_batch_strided_async(
    _q: *mut c_void,
    _ta: i32,
    _tb: i32,
    _m: i64,
    _n: i64,
    _k: i64,
    _alpha: f32,
    _a: *const u16,
    _lda: i64,
    _stride_a: i64,
    _b: *const u16,
    _ldb: i64,
    _stride_b: i64,
    _beta: f32,
    _c: *mut f32,
    _ldc: i64,
    _stride_c: i64,
    _batch: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}
