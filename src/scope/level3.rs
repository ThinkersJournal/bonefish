//! The non-blocking level 3 operations on [`Scope`] beyond `gemm`: `symm`, `syrk`, `syr2k`,
//! `trmm`, `trsm`. Same contract as the other scope operations: extents are checked in Rust, and a
//! buffer an operation writes is distinct from the ones it reads.

use super::{Event, NO_DEPS, Operand, Scope, Write, distinct};
use crate::blas::{Transpose, check_matrix, dim};
use crate::error::Result;
use crate::level2::{Diag, Uplo};
use crate::level3::{Level3, Side};

/// Rows and columns of the special matrix `A` of `symm`, `trmm` and `trsm`: `m x m` on the left,
/// `n x n` on the right.
fn side_dim(side: Side, m: usize, n: usize) -> usize {
    match side {
        Side::Left => m,
        Side::Right => n,
    }
}

/// Stored shape of `op(A)`-operand of `syrk` and `syr2k`: `n x k` without transposition, `k x n`
/// with.
fn rank_k_shape(trans: Transpose, n: usize, k: usize) -> (usize, usize) {
    if trans == Transpose::No {
        (n, k)
    } else {
        (k, n)
    }
}

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// `C = alpha * A * B + beta * C` (`Side::Left`) or `alpha * B * A + beta * C` (`Side::Right`); `A` symmetric, only the `uplo` triangle read; `B` and `C` are `m x n`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
    /// output buffer is also an input; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn symm<T: Level3>(
        &'scope self,
        side: Side,
        uplo: Uplo,
        m: usize,
        n: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        b: &impl Operand<T>,
        ldb: usize,
        beta: T,
        c: &Write<'scope, T>,
        ldc: usize,
    ) -> Result<Event<'scope>> {
        let (a, b, c) = (a.buffer(), b.buffer(), &*c.buf);
        let na = side_dim(side, m, n);
        check_matrix("A", a.len(), lda, na, na)?;
        check_matrix("B", b.len(), ldb, m, n)?;
        check_matrix("C", c.len(), ldc, m, n)?;
        distinct(c, "C", a, "A")?;
        distinct(c, "C", b, "B")?;
        // SAFETY: extents checked; USM of this scope's queue, borrowed until the scope ends,
        // which waits first; the written buffer is distinct from the read ones.
        let raw = unsafe {
            T::symm_raw_async(
                self.queue.raw(),
                side.code(),
                uplo.code(),
                dim("m", m)?,
                dim("n", n)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                b.raw(),
                dim("ldb", ldb)?,
                beta,
                c.raw(),
                dim("ldc", ldc)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `C = alpha * op(A) * op(A)^T + beta * C`, `C` symmetric `n x n`, only the `uplo` triangle updated; `op(A)` is `n x k`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
    /// output buffer is also an input; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn syrk<T: Level3>(
        &'scope self,
        uplo: Uplo,
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        beta: T,
        c: &Write<'scope, T>,
        ldc: usize,
    ) -> Result<Event<'scope>> {
        let (a, c) = (a.buffer(), &*c.buf);
        let (ar, ac) = rank_k_shape(trans, n, k);
        check_matrix("A", a.len(), lda, ar, ac)?;
        check_matrix("C", c.len(), ldc, n, n)?;
        distinct(c, "C", a, "A")?;
        // SAFETY: extents checked; USM of this scope's queue, borrowed until the scope ends,
        // which waits first; the written buffer is distinct from the read ones.
        let raw = unsafe {
            T::syrk_raw_async(
                self.queue.raw(),
                uplo.code(),
                trans.code(),
                dim("n", n)?,
                dim("k", k)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                beta,
                c.raw(),
                dim("ldc", ldc)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `C = alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C`, `C` symmetric `n x n`, only the `uplo` triangle updated; `op(A)` and `op(B)` are `n x k`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
    /// output buffer is also an input; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn syr2k<T: Level3>(
        &'scope self,
        uplo: Uplo,
        trans: Transpose,
        n: usize,
        k: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        b: &impl Operand<T>,
        ldb: usize,
        beta: T,
        c: &Write<'scope, T>,
        ldc: usize,
    ) -> Result<Event<'scope>> {
        let (a, b, c) = (a.buffer(), b.buffer(), &*c.buf);
        let (r, cl) = rank_k_shape(trans, n, k);
        check_matrix("A", a.len(), lda, r, cl)?;
        check_matrix("B", b.len(), ldb, r, cl)?;
        check_matrix("C", c.len(), ldc, n, n)?;
        distinct(c, "C", a, "A")?;
        distinct(c, "C", b, "B")?;
        // SAFETY: extents checked; USM of this scope's queue, borrowed until the scope ends,
        // which waits first; the written buffer is distinct from the read ones.
        let raw = unsafe {
            T::syr2k_raw_async(
                self.queue.raw(),
                uplo.code(),
                trans.code(),
                dim("n", n)?,
                dim("k", k)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                b.raw(),
                dim("ldb", ldb)?,
                beta,
                c.raw(),
                dim("ldc", ldc)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `B = alpha * op(A) * B` (`Side::Left`) or `alpha * B * op(A)` (`Side::Right`); `A` triangular, `B` is `m x n`, overwritten.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
    /// output buffer is also an input; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn trmm<T: Level3>(
        &'scope self,
        side: Side,
        uplo: Uplo,
        trans: Transpose,
        diag: Diag,
        m: usize,
        n: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        b: &Write<'scope, T>,
        ldb: usize,
    ) -> Result<Event<'scope>> {
        let (a, b) = (a.buffer(), &*b.buf);
        let na = side_dim(side, m, n);
        check_matrix("A", a.len(), lda, na, na)?;
        check_matrix("B", b.len(), ldb, m, n)?;
        distinct(b, "B", a, "A")?;
        // SAFETY: extents checked; USM of this scope's queue, borrowed until the scope ends,
        // which waits first; the written buffer is distinct from the read ones.
        let raw = unsafe {
            T::trmm_raw_async(
                self.queue.raw(),
                side.code(),
                uplo.code(),
                trans.code(),
                diag.code(),
                dim("m", m)?,
                dim("n", n)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                b.raw(),
                dim("ldb", ldb)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// Solve `op(A) * X = alpha * B` (`Side::Left`) or `X * op(A) = alpha * B` (`Side::Right`) for `X`, overwriting `B` (`m x n`); `A` triangular, not checked for singularity.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if an extent does not fit or the
    /// output buffer is also an input; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn trsm<T: Level3>(
        &'scope self,
        side: Side,
        uplo: Uplo,
        trans: Transpose,
        diag: Diag,
        m: usize,
        n: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        b: &Write<'scope, T>,
        ldb: usize,
    ) -> Result<Event<'scope>> {
        let (a, b) = (a.buffer(), &*b.buf);
        let na = side_dim(side, m, n);
        check_matrix("A", a.len(), lda, na, na)?;
        check_matrix("B", b.len(), ldb, m, n)?;
        distinct(b, "B", a, "A")?;
        // SAFETY: extents checked; USM of this scope's queue, borrowed until the scope ends,
        // which waits first; the written buffer is distinct from the read ones.
        let raw = unsafe {
            T::trsm_raw_async(
                self.queue.raw(),
                side.code(),
                uplo.code(),
                trans.code(),
                diag.code(),
                dim("m", m)?,
                dim("n", n)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                b.raw(),
                dim("ldb", ldb)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }
}
