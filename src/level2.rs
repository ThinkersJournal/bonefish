//! Level 2 BLAS beyond `gemv` (R3): `rotm`, `symv`, `syr`, `syr2`, `trmv`, `trsv`, `gbmv`, for
//! `f32` and `f64`. The non-blocking forms are on [`Scope`](crate::Scope); the blocking forms are
//! here and in [`blas`](crate::blas). Complex Hermitian counterparts are not wrapped yet.

use std::ffi::c_void;

use bonefish_sys as sys;

use crate::error::Result;
use crate::layout::Transpose;
use crate::queue::Queue;
use crate::scalar::Scalar;
use crate::usm::UsmBuffer;

/// Which triangle of a symmetric or triangular matrix is stored and read. The other triangle is
/// never touched.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Uplo {
    /// The upper triangle.
    Upper,
    /// The lower triangle.
    Lower,
}

impl Uplo {
    pub(crate) fn code(self) -> i32 {
        match self {
            Uplo::Upper => 0,
            Uplo::Lower => 1,
        }
    }
}

/// Whether a triangular matrix has an implicit unit diagonal (the stored diagonal is not read).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Diag {
    /// The stored diagonal is used.
    NonUnit,
    /// The diagonal is taken to be all ones.
    Unit,
}

impl Diag {
    pub(crate) fn code(self) -> i32 {
        match self {
            Diag::NonUnit => 0,
            Diag::Unit => 1,
        }
    }
}

/// A real type the level 2 functions in this module accept: `f32` or `f64`. Sealed through
/// [`Element`](crate::Element).
pub trait Level2: Scalar {
    /// # Safety
    /// Pointers are USM from `q` covering the stated extents, alive and untouched by the host
    /// until the returned event (or the queue) has been waited on; `param` holds 5 elements.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn rotm_raw_async(
        q: *mut c_void,
        n: i64,
        x: *mut Self,
        incx: i64,
        y: *mut Self,
        incy: i64,
        param: *mut Self,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// As [`Level2::rotm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn symv_raw_async(
        q: *mut c_void,
        uplo: i32,
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

    /// # Safety
    /// As [`Level2::rotm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syr_raw_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: Self,
        x: *const Self,
        incx: i64,
        a: *mut Self,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// As [`Level2::rotm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syr2_raw_async(
        q: *mut c_void,
        uplo: i32,
        n: i64,
        alpha: Self,
        x: *const Self,
        incx: i64,
        y: *const Self,
        incy: i64,
        a: *mut Self,
        lda: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// As [`Level2::rotm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trmv_raw_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const Self,
        lda: i64,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// As [`Level2::rotm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn trsv_raw_async(
        q: *mut c_void,
        uplo: i32,
        trans: i32,
        diag: i32,
        n: i64,
        a: *const Self,
        lda: i64,
        x: *mut Self,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;

    /// # Safety
    /// As [`Level2::rotm_raw_async`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gbmv_raw_async(
        q: *mut c_void,
        trans: i32,
        m: i64,
        n: i64,
        kl: i64,
        ku: i64,
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

macro_rules! impl_level2 {
    ($t:ty, $rotm:ident, $symv:ident, $syr:ident, $syr2:ident, $trmv:ident, $trsv:ident, $gbmv:ident) => {
        impl Level2 for $t {
            unsafe fn rotm_raw_async(
                q: *mut c_void,
                n: i64,
                x: *mut Self,
                incx: i64,
                y: *mut Self,
                incy: i64,
                param: *mut Self,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe { sys::$rotm(q, n, x, incx, y, incy, param, deps, n_deps) }
            }

            unsafe fn symv_raw_async(
                q: *mut c_void,
                uplo: i32,
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
                    sys::$symv(
                        q, uplo, n, alpha, a, lda, x, incx, beta, y, incy, deps, n_deps,
                    )
                }
            }

            unsafe fn syr_raw_async(
                q: *mut c_void,
                uplo: i32,
                n: i64,
                alpha: Self,
                x: *const Self,
                incx: i64,
                a: *mut Self,
                lda: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe { sys::$syr(q, uplo, n, alpha, x, incx, a, lda, deps, n_deps) }
            }

            unsafe fn syr2_raw_async(
                q: *mut c_void,
                uplo: i32,
                n: i64,
                alpha: Self,
                x: *const Self,
                incx: i64,
                y: *const Self,
                incy: i64,
                a: *mut Self,
                lda: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe { sys::$syr2(q, uplo, n, alpha, x, incx, y, incy, a, lda, deps, n_deps) }
            }

            unsafe fn trmv_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                diag: i32,
                n: i64,
                a: *const Self,
                lda: i64,
                x: *mut Self,
                incx: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe { sys::$trmv(q, uplo, trans, diag, n, a, lda, x, incx, deps, n_deps) }
            }

            unsafe fn trsv_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                diag: i32,
                n: i64,
                a: *const Self,
                lda: i64,
                x: *mut Self,
                incx: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe { sys::$trsv(q, uplo, trans, diag, n, a, lda, x, incx, deps, n_deps) }
            }

            unsafe fn gbmv_raw_async(
                q: *mut c_void,
                trans: i32,
                m: i64,
                n: i64,
                kl: i64,
                ku: i64,
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
                    sys::$gbmv(
                        q, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy, deps, n_deps,
                    )
                }
            }
        }
    };
}

impl_level2!(
    f32,
    bf_srotm_async,
    bf_ssymv_async,
    bf_ssyr_async,
    bf_ssyr2_async,
    bf_strmv_async,
    bf_strsv_async,
    bf_sgbmv_async
);
impl_level2!(
    f64,
    bf_drotm_async,
    bf_dsymv_async,
    bf_dsyr_async,
    bf_dsyr2_async,
    bf_dtrmv_async,
    bf_dtrsv_async,
    bf_dgbmv_async
);

/// Apply the modified Givens rotation `param` to `x` and `y` in place. `param` is the 5-element
/// LAPACK/BLAS `rotm` parameter (flag, `h11`, `h21`, `h12`, `h22`). Blocking.
///
/// # Errors
///
/// [`Error::QueueMismatch`](crate::Error::QueueMismatch) if a buffer is from another queue;
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit, `param`
/// has fewer than 5 elements, or `x` and `y` are the same buffer; device errors otherwise.
pub fn rotm<T: Level2>(
    queue: &Queue,
    n: usize,
    x: &mut UsmBuffer<T>,
    incx: usize,
    y: &mut UsmBuffer<T>,
    incy: usize,
    param: &UsmBuffer<T>,
) -> Result<()> {
    same_queue(queue, &[x.queue(), y.queue(), param.queue()])?;
    queue.scope(|s| {
        let (param, x, y) = (s.read(param)?, s.write(x)?, s.write(y)?);
        s.rotm(n, &x, incx, &y, incy, &param)?;
        Ok(())
    })
}

/// `y = alpha * A * x + beta * y` with `A` symmetric `n x n`, only the `uplo` triangle read.
/// Blocking.
///
/// # Errors
///
/// As [`rotm`].
#[allow(clippy::too_many_arguments)]
pub fn symv<T: Level2>(
    queue: &Queue,
    uplo: Uplo,
    n: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    x: &UsmBuffer<T>,
    incx: usize,
    beta: T,
    y: &mut UsmBuffer<T>,
    incy: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), x.queue(), y.queue()])?;
    queue.scope(|s| {
        let (a, x, y) = (s.read(a)?, s.read(x)?, s.write(y)?);
        s.symv(uplo, n, alpha, &a, lda, &x, incx, beta, &y, incy)?;
        Ok(())
    })
}

/// `A = alpha * x * x^T + A` with `A` symmetric `n x n`, only the `uplo` triangle updated.
/// Blocking.
///
/// # Errors
///
/// As [`rotm`].
#[allow(clippy::too_many_arguments)]
pub fn syr<T: Level2>(
    queue: &Queue,
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &UsmBuffer<T>,
    incx: usize,
    a: &mut UsmBuffer<T>,
    lda: usize,
) -> Result<()> {
    same_queue(queue, &[x.queue(), a.queue()])?;
    queue.scope(|s| {
        let (x, a) = (s.read(x)?, s.write(a)?);
        s.syr(uplo, n, alpha, &x, incx, &a, lda)?;
        Ok(())
    })
}

/// `A = alpha * (x * y^T + y * x^T) + A` with `A` symmetric `n x n`, only the `uplo` triangle
/// updated. Blocking.
///
/// # Errors
///
/// As [`rotm`].
#[allow(clippy::too_many_arguments)]
pub fn syr2<T: Level2>(
    queue: &Queue,
    uplo: Uplo,
    n: usize,
    alpha: T,
    x: &UsmBuffer<T>,
    incx: usize,
    y: &UsmBuffer<T>,
    incy: usize,
    a: &mut UsmBuffer<T>,
    lda: usize,
) -> Result<()> {
    same_queue(queue, &[x.queue(), y.queue(), a.queue()])?;
    queue.scope(|s| {
        let (x, y, a) = (s.read(x)?, s.read(y)?, s.write(a)?);
        s.syr2(uplo, n, alpha, &x, incx, &y, incy, &a, lda)?;
        Ok(())
    })
}

/// `x = op(A) * x` with `A` triangular `n x n`. Blocking.
///
/// # Errors
///
/// As [`rotm`].
#[allow(clippy::too_many_arguments)]
pub fn trmv<T: Level2>(
    queue: &Queue,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &UsmBuffer<T>,
    lda: usize,
    x: &mut UsmBuffer<T>,
    incx: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), x.queue()])?;
    queue.scope(|s| {
        let (a, x) = (s.read(a)?, s.write(x)?);
        s.trmv(uplo, trans, diag, n, &a, lda, &x, incx)?;
        Ok(())
    })
}

/// Solve `op(A) * z = x` for `z`, overwriting `x`, with `A` triangular `n x n`. Blocking. A zero
/// on the diagonal gives non-finite results, as in reference BLAS; nothing is checked.
///
/// # Errors
///
/// As [`rotm`].
#[allow(clippy::too_many_arguments)]
pub fn trsv<T: Level2>(
    queue: &Queue,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    n: usize,
    a: &UsmBuffer<T>,
    lda: usize,
    x: &mut UsmBuffer<T>,
    incx: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), x.queue()])?;
    queue.scope(|s| {
        let (a, x) = (s.read(a)?, s.write(x)?);
        s.trsv(uplo, trans, diag, n, &a, lda, &x, incx)?;
        Ok(())
    })
}

/// `y = alpha * op(A) * x + beta * y` with `A` an `m x n` band matrix with `kl` sub- and `ku`
/// super-diagonals in BLAS band storage (`lda >= kl + ku + 1`). Blocking.
///
/// # Errors
///
/// As [`rotm`].
#[allow(clippy::too_many_arguments)]
pub fn gbmv<T: Level2>(
    queue: &Queue,
    trans: Transpose,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    x: &UsmBuffer<T>,
    incx: usize,
    beta: T,
    y: &mut UsmBuffer<T>,
    incy: usize,
) -> Result<()> {
    same_queue(queue, &[a.queue(), x.queue(), y.queue()])?;
    queue.scope(|s| {
        let (a, x, y) = (s.read(a)?, s.read(x)?, s.write(y)?);
        s.gbmv(
            trans, m, n, kl, ku, alpha, &a, lda, &x, incx, beta, &y, incy,
        )?;
        Ok(())
    })
}

pub(crate) fn same_queue(queue: &Queue, buffers: &[&Queue]) -> Result<()> {
    if buffers.iter().all(|q| queue.same_as(q)) {
        Ok(())
    } else {
        Err(crate::error::Error::QueueMismatch)
    }
}
