//! The element and scalar types the safe layer accepts. Both traits are sealed: each impl is
//! backed by a matching shim function, so an outside type could not be dispatched.

use std::ffi::c_void;

use bonefish_sys as sys;

mod sealed {
    pub trait Sealed {}
}

/// A complex number with the interleaved `(re, im)` layout oneMKL expects.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T> Complex<T> {
    /// `re + im i`.
    pub const fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }
}

/// A type that can live in a [`UsmBuffer`](crate::UsmBuffer).
pub trait Element: Copy + 'static + sealed::Sealed {
    /// The additive identity, used to zero-fill.
    const ZERO: Self;
}

macro_rules! element {
    ($t:ty, $zero:expr) => {
        impl sealed::Sealed for $t {}
        impl Element for $t {
            const ZERO: Self = $zero;
        }
    };
}
element!(f32, 0.0);
element!(f64, 0.0);
// Index results of `iamax`.
element!(i64, 0);

// Complex numbers over any real scalar. A blanket impl, not one per type, so generic code over
// `T: Scalar` can name `UsmBuffer<Complex<T>>`.
impl<T: Scalar> sealed::Sealed for Complex<T> {}
impl<T: Scalar> Element for Complex<T> {
    const ZERO: Self = Complex::new(T::ZERO, T::ZERO);
}

/// A real floating-point scalar: `f32` or `f64`. Dispatches to the matching shim function.
pub trait Scalar: Element {
    /// `0` single precision, `1` double (the shim's `prec` code).
    #[doc(hidden)]
    const PREC: i32;

    /// # Safety
    /// Pointers must be USM reachable from `q` and cover the extents implied by the dimensions
    /// and leading dimensions; the caller (this crate) has bounds-checked them.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
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

    /// Non-blocking `gemm`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm`], and in addition the operands must stay alive, and `c` untouched by
    /// the host, until the returned event (or the queue) has been waited on. `deps` points at
    /// `n_deps` live event handles from this queue.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_async(
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

    /// Non-blocking `axpy`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn axpy_async(
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

    /// Non-blocking `scal`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn scal_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `copy`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn copy_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `swap`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn swap_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `dot`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn dot_async(
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

    /// Non-blocking `nrm2`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn nrm2_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `asum`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn asum_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `iamax`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn iamax_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// Non-blocking `gemv`: returns the event handle, or null on failure (see `bf_last_error`).
    ///
    /// # Safety
    /// As [`Scalar::gemm_async`]: pointers are USM from `q` covering the stated extents, and stay
    /// alive, and untouched by the host, until the event or the queue has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemv_async(
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

    /// Non-blocking complex-to-complex DFT of this precision (`dir` 0 forward, 1 backward).
    ///
    /// # Safety
    /// `h` is a live plan of this precision from `bf_dft_c2c_new`; `input`/`out` are USM complex
    /// data covering the plan's length, alive and untouched by the host until the event or the
    /// queue has been waited on. `out` is null for an in-place plan.
    #[doc(hidden)]
    unsafe fn dft_compute_async(
        h: *mut c_void,
        dir: i32,
        input: *mut c_void,
        out: *mut c_void,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}

impl Scalar for f32 {
    const PREC: i32 = 0;
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_sgemm(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe {
            sys::bf_sgemm_async(
                q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn axpy_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_saxpy_async(q, n, alpha, x, incx, y, incy, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn scal_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_sscal_async(q, n, alpha, x, incx, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn copy_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_scopy_async(q, n, x, incx, y, incy, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn swap_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_sswap_async(q, n, x, incx, y, incy, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn dot_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_sdot_async(q, n, x, incx, y, incy, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn nrm2_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_snrm2_async(q, n, x, incx, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn asum_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_sasum_async(q, n, x, incx, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn iamax_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_siamax_async(q, n, x, incx, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemv_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe {
            sys::bf_sgemv_async(
                q, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, deps, n_deps,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn dft_compute_async(
        h: *mut c_void,
        dir: i32,
        input: *mut c_void,
        out: *mut c_void,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dft_compute_async(h, dir, input, out, deps, n_deps) }
    }
}

impl Scalar for f64 {
    const PREC: i32 = 1;
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dgemm(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe {
            sys::bf_dgemm_async(
                q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn axpy_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_daxpy_async(q, n, alpha, x, incx, y, incy, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn scal_async(
        q: *mut c_void,
        n: i64,
        alpha: Self,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dscal_async(q, n, alpha, x, incx, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn copy_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dcopy_async(q, n, x, incx, y, incy, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn swap_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dswap_async(q, n, x, incx, y, incy, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn dot_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_ddot_async(q, n, x, incx, y, incy, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn nrm2_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dnrm2_async(q, n, x, incx, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn asum_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dasum_async(q, n, x, incx, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn iamax_async(
        q: *mut c_void,
        n: i64,
        x: *const Self,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_diamax_async(q, n, x, incx, result, deps, n_deps) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemv_async(
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
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe {
            sys::bf_dgemv_async(
                q, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, deps, n_deps,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn dft_compute_async(
        h: *mut c_void,
        dir: i32,
        input: *mut c_void,
        out: *mut c_void,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void {
        // SAFETY: forwarded from the caller's contract on the trait method.
        unsafe { sys::bf_dft_compute_async(h, dir, input, out, deps, n_deps) }
    }
}
