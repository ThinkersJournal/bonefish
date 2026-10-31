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
}

macro_rules! scalar {
    ($t:ty, $prec:expr, $gemm:ident, $gemm_async:ident) => {
        impl Scalar for $t {
            const PREC: i32 = $prec;
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
                // SAFETY: forwarded from the caller's contract above.
                unsafe { sys::$gemm(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
            }
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
                // SAFETY: forwarded from the caller's contract above.
                unsafe {
                    sys::$gemm_async(
                        q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps,
                    )
                }
            }
        }
    };
}
scalar!(f32, 0, bf_sgemm, bf_sgemm_async);
scalar!(f64, 1, bf_dgemm, bf_dgemm_async);
