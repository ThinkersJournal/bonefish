//! The non-blocking level 2 operations on [`Scope`] beyond `gemv`: `rotm`, `symv`, `syr`, `syr2`,
//! `trmv`, `trsv`, `gbmv`. Same contract as the other scope operations: extents are checked in
//! Rust, and a buffer an operation writes is distinct from the ones it reads.

use super::{Event, NO_DEPS, Operand, Scope, Write, check_vec, distinct};
use crate::error::{Error, Result};
use crate::layout::{Transpose, check_matrix, dim};
use crate::level2::{Diag, Level2, Uplo};

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// Apply the modified Givens rotation `param` (5 elements: flag, `h11`, `h21`, `h12`, `h22`)
    /// to `x` and `y` in place.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit, `param` has fewer than 5 elements, or
    /// `x` and `y` (or `param`) are the same buffer; submission errors as for [`Scope::gemm`].
    pub fn rotm<T: Level2>(
        &'scope self,
        n: usize,
        x: &Write<'scope, T>,
        incx: usize,
        y: &Write<'scope, T>,
        incy: usize,
        param: &impl Operand<T>,
    ) -> Result<Event<'scope>> {
        let (x, y, param) = (&*x.buf, &*y.buf, param.buffer());
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        if param.len() < 5 {
            return Err(Error::InvalidArgument(format!(
                "param: rotm needs 5 elements, buffer has {}",
                param.len()
            )));
        }
        distinct(x, "x", y, "y")?;
        distinct(x, "x", param, "param")?;
        distinct(y, "y", param, "param")?;
        // SAFETY: extents checked; USM of this scope's queue, borrowed until the scope ends,
        // which waits first; `x`, `y` and `param` are distinct. `param` is only read.
        let raw = unsafe {
            T::rotm_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                param.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `y = alpha * A * x + beta * y`, `A` symmetric `n x n`, only the `uplo` triangle read.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `y` is the same buffer as `a` or
    /// `x`; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn symv<T: Level2>(
        &'scope self,
        uplo: Uplo,
        n: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        x: &impl Operand<T>,
        incx: usize,
        beta: T,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (a, x, y) = (a.buffer(), x.buffer(), &*y.buf);
        check_matrix("A", a.len(), lda, n, n)?;
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        distinct(y, "y", a, "A")?;
        distinct(y, "y", x, "x")?;
        // SAFETY: as `rotm`.
        let raw = unsafe {
            T::symv_raw_async(
                self.queue.raw(),
                uplo.code(),
                dim("n", n)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                x.raw(),
                dim("incx", incx)?,
                beta,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `A = alpha * x * x^T + A`, `A` symmetric `n x n`, only the `uplo` triangle updated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `a` is the same buffer as `x`;
    /// submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn syr<T: Level2>(
        &'scope self,
        uplo: Uplo,
        n: usize,
        alpha: T,
        x: &impl Operand<T>,
        incx: usize,
        a: &Write<'scope, T>,
        lda: usize,
    ) -> Result<Event<'scope>> {
        let (x, a) = (x.buffer(), &*a.buf);
        check_vec("x", x.len(), n, incx)?;
        check_matrix("A", a.len(), lda, n, n)?;
        distinct(a, "A", x, "x")?;
        // SAFETY: as `rotm`.
        let raw = unsafe {
            T::syr_raw_async(
                self.queue.raw(),
                uplo.code(),
                dim("n", n)?,
                alpha,
                x.raw(),
                dim("incx", incx)?,
                a.raw(),
                dim("lda", lda)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `A = alpha * (x * y^T + y * x^T) + A`, `A` symmetric `n x n`, only the `uplo` triangle
    /// updated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `a` is the same buffer as `x` or
    /// `y`; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn syr2<T: Level2>(
        &'scope self,
        uplo: Uplo,
        n: usize,
        alpha: T,
        x: &impl Operand<T>,
        incx: usize,
        y: &impl Operand<T>,
        incy: usize,
        a: &Write<'scope, T>,
        lda: usize,
    ) -> Result<Event<'scope>> {
        let (x, y, a) = (x.buffer(), y.buffer(), &*a.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        check_matrix("A", a.len(), lda, n, n)?;
        distinct(a, "A", x, "x")?;
        distinct(a, "A", y, "y")?;
        // SAFETY: as `rotm`.
        let raw = unsafe {
            T::syr2_raw_async(
                self.queue.raw(),
                uplo.code(),
                dim("n", n)?,
                alpha,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                a.raw(),
                dim("lda", lda)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `x = op(A) * x`, `A` triangular `n x n`, in place.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `x` is the same buffer as `a`;
    /// submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn trmv<T: Level2>(
        &'scope self,
        uplo: Uplo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &impl Operand<T>,
        lda: usize,
        x: &Write<'scope, T>,
        incx: usize,
    ) -> Result<Event<'scope>> {
        let (a, x) = (a.buffer(), &*x.buf);
        check_matrix("A", a.len(), lda, n, n)?;
        check_vec("x", x.len(), n, incx)?;
        distinct(x, "x", a, "A")?;
        // SAFETY: as `rotm`.
        let raw = unsafe {
            T::trmv_raw_async(
                self.queue.raw(),
                uplo.code(),
                trans.code(),
                diag.code(),
                dim("n", n)?,
                a.raw(),
                dim("lda", lda)?,
                x.raw(),
                dim("incx", incx)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// Solve `op(A) * z = x`, `A` triangular `n x n`, overwriting `x` with `z`. A zero on the
    /// diagonal gives non-finite results; nothing is checked.
    ///
    /// # Errors
    ///
    /// As [`Scope::trmv`].
    #[allow(clippy::too_many_arguments)]
    pub fn trsv<T: Level2>(
        &'scope self,
        uplo: Uplo,
        trans: Transpose,
        diag: Diag,
        n: usize,
        a: &impl Operand<T>,
        lda: usize,
        x: &Write<'scope, T>,
        incx: usize,
    ) -> Result<Event<'scope>> {
        let (a, x) = (a.buffer(), &*x.buf);
        check_matrix("A", a.len(), lda, n, n)?;
        check_vec("x", x.len(), n, incx)?;
        distinct(x, "x", a, "A")?;
        // SAFETY: as `rotm`.
        let raw = unsafe {
            T::trsv_raw_async(
                self.queue.raw(),
                uplo.code(),
                trans.code(),
                diag.code(),
                dim("n", n)?,
                a.raw(),
                dim("lda", lda)?,
                x.raw(),
                dim("incx", incx)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `y = alpha * op(A) * x + beta * y`, `A` an `m x n` band matrix with `kl` sub- and `ku`
    /// super-diagonals in BLAS band storage: `lda >= kl + ku + 1`, `n` columns.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `y` is the same buffer as `a` or
    /// `x`; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gbmv<T: Level2>(
        &'scope self,
        trans: Transpose,
        m: usize,
        n: usize,
        kl: usize,
        ku: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        x: &impl Operand<T>,
        incx: usize,
        beta: T,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (a, x, y) = (a.buffer(), x.buffer(), &*y.buf);
        let band = kl
            .checked_add(ku)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| Error::InvalidArgument("kl + ku + 1 overflows".into()))?;
        let (x_len, y_len) = if trans == Transpose::No {
            (n, m)
        } else {
            (m, n)
        };
        check_matrix("A", a.len(), lda, band, n)?;
        check_vec("x", x.len(), x_len, incx)?;
        check_vec("y", y.len(), y_len, incy)?;
        distinct(y, "y", a, "A")?;
        distinct(y, "y", x, "x")?;
        // SAFETY: as `rotm`.
        let raw = unsafe {
            T::gbmv_raw_async(
                self.queue.raw(),
                trans.code(),
                dim("m", m)?,
                dim("n", n)?,
                dim("kl", kl)?,
                dim("ku", ku)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                x.raw(),
                dim("incx", incx)?,
                beta,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }
}
