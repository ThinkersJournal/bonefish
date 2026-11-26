// SPDX-License-Identifier: MIT OR Apache-2.0
//! `BlasElement`: the element types level 1 and `gemv` accept (split out of `scalar.rs`, R15).

use std::ffi::c_void;

use bonefish_sys as sys;

use super::{Complex, Element, Scalar};

/// An element type the level 1 and `gemv` functions accept: `f32`, `f64`, `Complex<f32>` and
/// `Complex<f64>`. [`BlasElement::Real`] is the type `nrm2` and `asum` produce (the element type
/// itself for real data, the matching real for complex data). The real types forward to the
/// [`Scalar`] methods; the complex types call the `c`/`z` shim functions, passing complex scalars by
/// pointer. Sealed through [`Element`].
pub trait BlasElement: Element {
    /// The real type underlying this element (`Self` for real types).
    type Real: Scalar;

    /// Non-blocking `axpy`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn axpy_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `scal`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn scal_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `copy`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn copy_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `swap`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn swap_raw_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `dot`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn dot_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `dotc`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn dotc_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `nrm2`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn nrm2_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `asum`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn asum_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `iamax`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn iamax_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `gemv`: returns the event handle, or null on failure.
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, alive and
    /// untouched by the host until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemv_raw_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        x: *const Self,
        incx: i64,
        beta: Self,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}

impl<T: Scalar> BlasElement for T {
    type Real = T;

    unsafe fn axpy_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::axpy_async(q, n, alpha, x, incx, y, incy, deps, n_deps) }
    }

    unsafe fn scal_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::scal_async(q, n, alpha, x, incx, deps, n_deps) }
    }

    unsafe fn copy_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::copy_async(q, n, x, incx, y, incy, deps, n_deps) }
    }

    unsafe fn swap_raw_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::swap_async(q, n, x, incx, y, incy, deps, n_deps) }
    }

    unsafe fn dot_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::dot_async(q, n, x, incx, y, incy, result, deps, n_deps) }
    }

    unsafe fn dotc_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::dot_async(q, n, x, incx, y, incy, result, deps, n_deps) }
    }

    unsafe fn nrm2_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::nrm2_async(q, n, x, incx, result, deps, n_deps) }
    }

    unsafe fn asum_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::asum_async(q, n, x, incx, result, deps, n_deps) }
    }

    unsafe fn iamax_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe { <T as Scalar>::iamax_async(q, n, x, incx, result, deps, n_deps) }
    }

    unsafe fn gemv_raw_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        x: *const Self,
        incx: i64,
        beta: Self,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract. For real data `dotc` is `dot`.
        unsafe {
            <T as Scalar>::gemv_async(
                q, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, deps, n_deps,
            )
        }
    }
}

impl BlasElement for Complex<f32> {
    type Real = f32;

    unsafe fn axpy_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_caxpy_async(
                q,
                n,
                (&raw const alpha).cast(),
                x.cast(),
                incx,
                y.cast(),
                incy,
                deps,
                n_deps,
            )
        }
    }

    unsafe fn scal_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_cscal_async(
                q,
                n,
                (&raw const alpha).cast(),
                x.cast(),
                incx,
                deps,
                n_deps,
            )
        }
    }

    unsafe fn copy_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_ccopy_async(q, n, x.cast(), incx, y.cast(), incy, deps, n_deps) }
    }

    unsafe fn swap_raw_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_cswap_async(q, n, x.cast(), incx, y.cast(), incy, deps, n_deps) }
    }

    unsafe fn dot_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_cdotu_async(
                q,
                n,
                x.cast(),
                incx,
                y.cast(),
                incy,
                result.cast(),
                deps,
                n_deps,
            )
        }
    }

    unsafe fn dotc_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_cdotc_async(
                q,
                n,
                x.cast(),
                incx,
                y.cast(),
                incy,
                result.cast(),
                deps,
                n_deps,
            )
        }
    }

    unsafe fn nrm2_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_cnrm2_async(q, n, x.cast(), incx, result, deps, n_deps) }
    }

    unsafe fn asum_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_casum_async(q, n, x.cast(), incx, result, deps, n_deps) }
    }

    unsafe fn iamax_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_ciamax_async(q, n, x.cast(), incx, result, deps, n_deps) }
    }

    unsafe fn gemv_raw_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        x: *const Self,
        incx: i64,
        beta: Self,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f32`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_cgemv_async(
                q,
                trans,
                m,
                n,
                (&raw const alpha).cast(),
                a.cast(),
                lda,
                x.cast(),
                incx,
                (&raw const beta).cast(),
                y.cast(),
                incy,
                deps,
                n_deps,
            )
        }
    }
}

impl BlasElement for Complex<f64> {
    type Real = f64;

    unsafe fn axpy_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_zaxpy_async(
                q,
                n,
                (&raw const alpha).cast(),
                x.cast(),
                incx,
                y.cast(),
                incy,
                deps,
                n_deps,
            )
        }
    }

    unsafe fn scal_raw_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_zscal_async(
                q,
                n,
                (&raw const alpha).cast(),
                x.cast(),
                incx,
                deps,
                n_deps,
            )
        }
    }

    unsafe fn copy_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_zcopy_async(q, n, x.cast(), incx, y.cast(), incy, deps, n_deps) }
    }

    unsafe fn swap_raw_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_zswap_async(q, n, x.cast(), incx, y.cast(), incy, deps, n_deps) }
    }

    unsafe fn dot_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_zdotu_async(
                q,
                n,
                x.cast(),
                incx,
                y.cast(),
                incy,
                result.cast(),
                deps,
                n_deps,
            )
        }
    }

    unsafe fn dotc_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_zdotc_async(
                q,
                n,
                x.cast(),
                incx,
                y.cast(),
                incy,
                result.cast(),
                deps,
                n_deps,
            )
        }
    }

    unsafe fn nrm2_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_znrm2_async(q, n, x.cast(), incx, result, deps, n_deps) }
    }

    unsafe fn asum_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self::Real,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_zasum_async(q, n, x.cast(), incx, result, deps, n_deps) }
    }

    unsafe fn iamax_raw_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe { sys::bf_ziamax_async(q, n, x.cast(), incx, result, deps, n_deps) }
    }

    unsafe fn gemv_raw_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        x: *const Self,
        incx: i64,
        beta: Self,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two `f64`;
        // scalars outlive the call (the shim reads them before it returns). The rest is the
        // caller's contract.
        unsafe {
            sys::bf_zgemv_async(
                q,
                trans,
                m,
                n,
                (&raw const alpha).cast(),
                a.cast(),
                lda,
                x.cast(),
                incx,
                (&raw const beta).cast(),
                y.cast(),
                incy,
                deps,
                n_deps,
            )
        }
    }
}
