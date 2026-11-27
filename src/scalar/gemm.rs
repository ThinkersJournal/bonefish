// SPDX-License-Identifier: MIT OR Apache-2.0
//! `GemmElement`: the element types `gemm` accepts (split out of `scalar.rs`, R15).

use std::ffi::c_void;

use bonefish_sys as sys;

use super::{Complex, Element, Scalar};

/// An element type `gemm` accepts: `f32`, `f64`, `Complex<f32>` and `Complex<f64>`. The real types
/// forward to the [`Scalar`] methods; the complex types call the `c`/`z` shim functions, passing
/// `alpha` and `beta` by pointer (a C++ `std::complex` is not passed by value across `extern "C"`).
/// Sealed through [`Element`].
pub trait GemmElement: Element {
    /// Blocking `gemm`; returns a shim status.
    ///
    /// # Safety
    /// As [`Scalar::gemm`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_raw(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) -> i32;

    /// Non-blocking `gemm`; returns the event handle or null.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_raw_async(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}

impl<T: Scalar> GemmElement for T {
    unsafe fn gemm_raw(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) -> i32 {
        // SAFETY: forwarded from the caller's contract.
        unsafe { <T as Scalar>::gemm(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
    }

    unsafe fn gemm_raw_async(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract.
        unsafe {
            <T as Scalar>::gemm_async(
                q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps,
            )
        }
    }
}

impl GemmElement for Complex<f32> {
    unsafe fn gemm_raw(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) -> i32 {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two
        // `f32`; `alpha` and `beta` outlive the call. The rest is the caller's contract.
        unsafe {
            sys::bf_cgemm(
                q,
                ta,
                tb,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&raw const beta).cast(),
                c.cast(),
                ldc,
            )
        }
    }

    unsafe fn gemm_raw_async(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: as `gemm_raw`; the shim reads `alpha` and `beta` before it returns.
        unsafe {
            sys::bf_cgemm_async(
                q,
                ta,
                tb,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&raw const beta).cast(),
                c.cast(),
                ldc,
                deps,
                n_deps,
            )
        }
    }
}

impl GemmElement for Complex<f64> {
    unsafe fn gemm_raw(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) -> i32 {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two
        // `f64`; `alpha` and `beta` outlive the call. The rest is the caller's contract.
        unsafe {
            sys::bf_zgemm(
                q,
                ta,
                tb,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&raw const beta).cast(),
                c.cast(),
                ldc,
            )
        }
    }

    unsafe fn gemm_raw_async(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: as `gemm_raw`; the shim reads `alpha` and `beta` before it returns.
        unsafe {
            sys::bf_zgemm_async(
                q,
                ta,
                tb,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&raw const beta).cast(),
                c.cast(),
                ldc,
                deps,
                n_deps,
            )
        }
    }
}
