//! Level 3 BLAS beyond `gemm` (R3): `symm`, `syrk`, `syr2k`, `trmm`, `trsm` for `f32`, `f64`,
//! `Complex<f32>` and `Complex<f64>`, and the Hermitian `hemm`, `herk`, `her2k` for the complex types.
//! The non-blocking forms are on [`Scope`](crate::Scope); the blocking forms are here. Batched and
//! half-precision variants are not wrapped yet. For complex types `syrk` and `syr2k` are the
//! complex-SYMMETRIC updates (transpose, not conjugate transpose).

use std::ffi::c_void;

use bonefish_sys as sys;

use crate::blas::Transpose;
use crate::error::Result;
use crate::level2::{Diag, Uplo, same_queue};
use crate::queue::Queue;
use crate::scalar::{Element, Scalar};
use crate::usm::UsmBuffer;

/// Which side the special matrix of `symm`, `trmm` and `trsm` multiplies from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    /// The special matrix `A` is on the left: `A * B`.
    Left,
    /// The special matrix `A` is on the right: `B * A`.
    Right,
}

impl Side {
    pub(crate) fn code(self) -> i32 {
        match self {
            Side::Left => 0,
            Side::Right => 1,
        }
    }
}

/// An element type the level 3 functions in this module accept: `f32`, `f64`, `Complex<f32>` or
/// `Complex<f64>`. Sealed through [`Element`].
pub trait Level3: Element {
    /// # Safety
    /// Pointers are USM from `q` covering the stated extents, alive and untouched by the host
    /// until the returned event (or the queue) has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn symm_raw_async(
        q: *mut c_void,
        side: i32,
        uplo: i32,
        m: i64,
        n: i64,
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

    /// # Safety
    /// Pointers are USM from `q` covering the stated extents, alive and untouched by the host
    /// until the returned event (or the queue) has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syrk_raw_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// Pointers are USM from `q` covering the stated extents, alive and untouched by the host
    /// until the returned event (or the queue) has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syr2k_raw_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
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

    /// # Safety
    /// Pointers are USM from `q` covering the stated extents, alive and untouched by the host
    /// until the returned event (or the queue) has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trmm_raw_async(
        q: *mut c_void,
        side: i32,
        uplo: i32,
        trans: i32,
        diag: i32,
        m: i64,
        n: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *mut Self,
        ldb: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// Pointers are USM from `q` covering the stated extents, alive and untouched by the host
    /// until the returned event (or the queue) has been waited on.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trsm_raw_async(
        q: *mut c_void,
        side: i32,
        uplo: i32,
        trans: i32,
        diag: i32,
        m: i64,
        n: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *mut Self,
        ldb: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}

/// A complex type the Hermitian level 3 functions accept: `Complex<f32>` or `Complex<f64>`.
/// `herk` and `her2k` take REAL `alpha` (`herk`) and `beta`, as BLAS defines them; `herk` and
/// `her2k` accept `Transpose::No` and `Transpose::ConjTrans` only (oneMKL rejects the other).
pub trait Hermitian3: Level3 {
    /// The real type underlying this element.
    type Real: Scalar;

    /// # Safety
    /// As for [`Level3::symm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn hemm_raw_async(
        q: *mut c_void,
        side: i32,
        uplo: i32,
        m: i64,
        n: i64,
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

    /// # Safety
    /// As for [`Level3::symm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn herk_raw_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        n: i64,
        k: i64,
        alpha: Self::Real,
        a: *const Self,
        lda: i64,
        beta: Self::Real,
        c: *mut Self,
        ldc: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// As for [`Level3::symm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn her2k_raw_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self::Real,
        c: *mut Self,
        ldc: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}

macro_rules! impl_level3 {
    ($t:ty, $symm:ident, $syrk:ident, $syr2k:ident, $trmm:ident, $trsm:ident) => {
        impl Level3 for $t {
            unsafe fn symm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                m: i64,
                n: i64,
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
                    sys::$symm(
                        q, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps,
                    )
                }
            }

            unsafe fn syrk_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe {
                    sys::$syrk(
                        q, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, deps, n_deps,
                    )
                }
            }

            unsafe fn syr2k_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
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
                    sys::$syr2k(
                        q, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps,
                    )
                }
            }

            unsafe fn trmm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                trans: i32,
                diag: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *mut Self,
                ldb: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe {
                    sys::$trmm(
                        q, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, deps, n_deps,
                    )
                }
            }

            unsafe fn trsm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                trans: i32,
                diag: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *mut Self,
                ldb: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe {
                    sys::$trsm(
                        q, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, deps, n_deps,
                    )
                }
            }
        }
    };
}

impl_level3!(
    f32,
    bf_ssymm_async,
    bf_ssyrk_async,
    bf_ssyr2k_async,
    bf_strmm_async,
    bf_strsm_async
);
impl_level3!(
    f64,
    bf_dsymm_async,
    bf_dsyrk_async,
    bf_dsyr2k_async,
    bf_dtrmm_async,
    bf_dtrsm_async
);

mod complex;

/// `C = alpha * A * B + beta * C` (`Side::Left`) or `alpha * B * A + beta * C` (`Side::Right`); `A` symmetric, only the `uplo` triangle read; `B` and `C` are `m x n`. Blocking.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
/// output buffer is also an input; [`Error::QueueMismatch`](crate::Error::QueueMismatch) if a
/// buffer belongs to another queue; submission errors as for [`gemm`](crate::blas::gemm).
#[allow(clippy::too_many_arguments)]
pub fn symm<T: Level3>(
    queue: &Queue,
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    b: &UsmBuffer<T>,
    ldb: usize,
    beta: T,
    c: &mut UsmBuffer<T>,
    ldc: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), b.queue(), c.queue()])?;
    queue.scope(|s| {
        let (a, b, c) = (s.read(a)?, s.read(b)?, s.write(c)?);
        s.symm(side, uplo, m, n, alpha, &a, lda, &b, ldb, beta, &c, ldc)?;
        Ok(())
    })
}

/// `C = alpha * op(A) * op(A)^T + beta * C`, `C` symmetric `n x n`, only the `uplo` triangle updated; `op(A)` is `n x k`. Blocking.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
/// output buffer is also an input; [`Error::QueueMismatch`](crate::Error::QueueMismatch) if a
/// buffer belongs to another queue; submission errors as for [`gemm`](crate::blas::gemm).
#[allow(clippy::too_many_arguments)]
pub fn syrk<T: Level3>(
    queue: &Queue,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    beta: T,
    c: &mut UsmBuffer<T>,
    ldc: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), c.queue()])?;
    queue.scope(|s| {
        let (a, c) = (s.read(a)?, s.write(c)?);
        s.syrk(uplo, trans, n, k, alpha, &a, lda, beta, &c, ldc)?;
        Ok(())
    })
}

/// `C = alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C`, `C` symmetric `n x n`, only the `uplo` triangle updated; `op(A)` and `op(B)` are `n x k`. Blocking.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
/// output buffer is also an input; [`Error::QueueMismatch`](crate::Error::QueueMismatch) if a
/// buffer belongs to another queue; submission errors as for [`gemm`](crate::blas::gemm).
#[allow(clippy::too_many_arguments)]
pub fn syr2k<T: Level3>(
    queue: &Queue,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    b: &UsmBuffer<T>,
    ldb: usize,
    beta: T,
    c: &mut UsmBuffer<T>,
    ldc: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), b.queue(), c.queue()])?;
    queue.scope(|s| {
        let (a, b, c) = (s.read(a)?, s.read(b)?, s.write(c)?);
        s.syr2k(uplo, trans, n, k, alpha, &a, lda, &b, ldb, beta, &c, ldc)?;
        Ok(())
    })
}

/// `B = alpha * op(A) * B` (`Side::Left`) or `alpha * B * op(A)` (`Side::Right`); `A` triangular, `B` is `m x n`, overwritten. Blocking.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
/// output buffer is also an input; [`Error::QueueMismatch`](crate::Error::QueueMismatch) if a
/// buffer belongs to another queue; submission errors as for [`gemm`](crate::blas::gemm).
#[allow(clippy::too_many_arguments)]
pub fn trmm<T: Level3>(
    queue: &Queue,
    side: Side,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    b: &mut UsmBuffer<T>,
    ldb: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), b.queue()])?;
    queue.scope(|s| {
        let (a, b) = (s.read(a)?, s.write(b)?);
        s.trmm(side, uplo, trans, diag, m, n, alpha, &a, lda, &b, ldb)?;
        Ok(())
    })
}

/// Solve `op(A) * X = alpha * B` (`Side::Left`) or `X * op(A) = alpha * B` (`Side::Right`) for `X`, overwriting `B` (`m x n`); `A` triangular, not checked for singularity. Blocking.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
/// output buffer is also an input; [`Error::QueueMismatch`](crate::Error::QueueMismatch) if a
/// buffer belongs to another queue; submission errors as for [`gemm`](crate::blas::gemm).
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: Level3>(
    queue: &Queue,
    side: Side,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    b: &mut UsmBuffer<T>,
    ldb: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), b.queue()])?;
    queue.scope(|s| {
        let (a, b) = (s.read(a)?, s.write(b)?);
        s.trsm(side, uplo, trans, diag, m, n, alpha, &a, lda, &b, ldb)?;
        Ok(())
    })
}

/// `C = alpha * A * B + beta * C` (`Side::Left`) or `alpha * B * A + beta * C` (`Side::Right`); `A` Hermitian, only the `uplo` triangle read; `B` and `C` are `m x n`. Blocking.
///
/// # Errors
///
/// As for [`symm`].
#[allow(clippy::too_many_arguments)]
pub fn hemm<T: Hermitian3>(
    queue: &Queue,
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    b: &UsmBuffer<T>,
    ldb: usize,
    beta: T,
    c: &mut UsmBuffer<T>,
    ldc: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), b.queue(), c.queue()])?;
    queue.scope(|s| {
        let (a, b, c) = (s.read(a)?, s.read(b)?, s.write(c)?);
        s.hemm(side, uplo, m, n, alpha, &a, lda, &b, ldb, beta, &c, ldc)?;
        Ok(())
    })
}

/// `C = alpha * op(A) * op(A)^H + beta * C`, `C` Hermitian `n x n`, only the `uplo` triangle updated; `alpha` and `beta` are real. Blocking.
///
/// # Errors
///
/// As for [`syrk`].
#[allow(clippy::too_many_arguments)]
pub fn herk<T: Hermitian3>(
    queue: &Queue,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T::Real,
    a: &UsmBuffer<T>,
    lda: usize,
    beta: T::Real,
    c: &mut UsmBuffer<T>,
    ldc: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), c.queue()])?;
    queue.scope(|s| {
        let (a, c) = (s.read(a)?, s.write(c)?);
        s.herk(uplo, trans, n, k, alpha, &a, lda, beta, &c, ldc)?;
        Ok(())
    })
}

/// `C = alpha * op(A) * op(B)^H + conj(alpha) * op(B) * op(A)^H + beta * C`, `C` Hermitian `n x n`, only the `uplo` triangle updated; `beta` is real. Blocking.
///
/// # Errors
///
/// As for [`syr2k`].
#[allow(clippy::too_many_arguments)]
pub fn her2k<T: Hermitian3>(
    queue: &Queue,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    b: &UsmBuffer<T>,
    ldb: usize,
    beta: T::Real,
    c: &mut UsmBuffer<T>,
    ldc: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), b.queue(), c.queue()])?;
    queue.scope(|s| {
        let (a, b, c) = (s.read(a)?, s.read(b)?, s.write(c)?);
        s.her2k(uplo, trans, n, k, alpha, &a, lda, &b, ldb, beta, &c, ldc)?;
        Ok(())
    })
}
