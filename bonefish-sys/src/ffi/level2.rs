// SPDX-License-Identifier: MIT OR Apache-2.0
//! Foreign declarations: level 2 BLAS. Split out of `lib.rs` (R15).

use std::ffi::c_void;

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sger(
        q: *mut c_void,
        m: i64,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        a: *mut f32,
        lda: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ssymv(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f32,
        a: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: f32,
        y: *mut f32,
        incy: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ssyr(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        a: *mut f32,
        lda: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ssyr2(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        a: *mut f32,
        lda: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_strmv(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f32,
        lda: i64,
        x: *mut f32,
        incx: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_strsv(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f32,
        lda: i64,
        x: *mut f32,
        incx: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sgbmv(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        kl: i64,
        ku: i64,
        alpha: f32,
        a: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: f32,
        y: *mut f32,
        incy: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dsymv(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: f64,
        y: *mut f64,
        incy: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dsyr(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        a: *mut f64,
        lda: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dsyr2(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        a: *mut f64,
        lda: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dtrmv(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f64,
        lda: i64,
        x: *mut f64,
        incx: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dtrsv(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f64,
        lda: i64,
        x: *mut f64,
        incx: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dgbmv(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        kl: i64,
        ku: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: f64,
        y: *mut f64,
        incy: i64,
    ) -> i32;
    /// `y = alpha * op(A) * x + beta * y`, `A` is `m x n` column-major. `trans`: 0 N, 1 T, 2 C.
    pub fn bf_sgemv(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: f32,
        a: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: f32,
        y: *mut f32,
        incy: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dger(
        q: *mut c_void,
        m: i64,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        a: *mut f64,
        lda: i64,
    ) -> i32;
    /// `y = alpha * op(A) * x + beta * y`, `A` is `m x n` column-major. `trans`: 0 N, 1 T, 2 C.
    pub fn bf_dgemv(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: f64,
        y: *mut f64,
        incy: i64,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sger_async(
        q: *mut c_void,
        m: i64,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        a: *mut f32,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ssymv_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f32,
        a: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: f32,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ssyr_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        a: *mut f32,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ssyr2_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        a: *mut f32,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_strmv_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f32,
        lda: i64,
        x: *mut f32,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_strsv_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f32,
        lda: i64,
        x: *mut f32,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sgbmv_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        kl: i64,
        ku: i64,
        alpha: f32,
        a: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: f32,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dsymv_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: f64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dsyr_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        a: *mut f64,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dsyr2_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        a: *mut f64,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dtrmv_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f64,
        lda: i64,
        x: *mut f64,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dtrsv_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const f64,
        lda: i64,
        x: *mut f64,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dgbmv_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        kl: i64,
        ku: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: f64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sgemv_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: f32,
        a: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: f32,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dger_async(
        q: *mut c_void,
        m: i64,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        a: *mut f64,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dgemv_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: f64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_cgemv_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: *const f32,
        a: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: *const f32,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zgemv_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: *const f64,
        a: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: *const f64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}
