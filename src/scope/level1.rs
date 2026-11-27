// SPDX-License-Identifier: MIT OR Apache-2.0
//! Level 1 and `gemv` on [`Scope`]: `axpy`, `scal`, `copy`, `swap`, `dot`, `dotc`, `nrm2`, `asum`,
//! `iamax`, `gemv`. Split out of `scope.rs` (R15).

use super::*;

/// Level 1 and `gemv`. Each call returns once the work is submitted. Operations on one queue
/// run in submission order because the shim chains each asynchronous submission to the queue's
/// previous one; an in-order queue alone does not order oneMKL calls (`ROADMAP.md` R3c). Vector
/// increments must be at least 1.
impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// `y = alpha * x + y`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit its buffer or `y` is the same buffer as
    /// `x`; submission errors as for [`Scope::gemm`].
    pub fn axpy<T: BlasElement>(
        &'scope self,
        n: usize,
        alpha: T,
        x: &impl Operand<T>,
        incx: usize,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (x, y) = (x.buffer(), &*y.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        distinct(y, "y", x, "x")?;
        // SAFETY: extents checked; buffers are USM of this scope's queue, borrowed until the scope
        // ends, which waits first; `y` is distinct from `x`.
        let raw = unsafe {
            T::axpy_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                alpha,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `x = alpha * x`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the extent does not fit; submission errors as for
    /// [`Scope::gemm`].
    pub fn scal<T: BlasElement>(
        &'scope self,
        n: usize,
        alpha: T,
        x: &Write<'scope, T>,
        incx: usize,
    ) -> Result<Event<'scope>> {
        let x = &*x.buf;
        check_vec("x", x.len(), n, incx)?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::scal_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                alpha,
                x.raw(),
                dim("incx", incx)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `y = x`.
    ///
    /// # Errors
    ///
    /// As [`Scope::axpy`].
    pub fn copy<T: BlasElement>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (x, y) = (x.buffer(), &*y.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        distinct(y, "y", x, "x")?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::copy_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// Exchange `x` and `y`.
    ///
    /// # Errors
    ///
    /// As [`Scope::axpy`].
    pub fn swap<T: BlasElement>(
        &'scope self,
        n: usize,
        x: &Write<'scope, T>,
        incx: usize,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (x, y) = (&*x.buf, &*y.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        distinct(x, "x", y, "y")?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::swap_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0] = x . y`, the plain (unconjugated) product `sum x_i * y_i`; for complex data this is
    /// BLAS `dotu`. See [`Scope::dotc`] for the conjugated one. The result is written when the work runs, so read it after the scope
    /// (or after waiting for the returned event).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `result` is the same buffer as `x`
    /// or `y`; submission errors as for [`Scope::gemm`].
    pub fn dot<T: BlasElement>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        y: &impl Operand<T>,
        incy: usize,
        result: &Write<'scope, T>,
    ) -> Result<Event<'scope>> {
        let (x, y, r) = (x.buffer(), y.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        distinct(r, "result", y, "y")?;
        // SAFETY: as `axpy`; `result` has at least one element and is distinct from the inputs.
        let raw = unsafe {
            T::dot_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0] = conj(x) . y`: the conjugated dot product. For real data this is `dot`. The result is written when the work runs, so read it after the scope
    /// (or after waiting for the returned event).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `result` is the same buffer as `x`
    /// or `y`; submission errors as for [`Scope::gemm`].
    pub fn dotc<T: BlasElement>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        y: &impl Operand<T>,
        incy: usize,
        result: &Write<'scope, T>,
    ) -> Result<Event<'scope>> {
        let (x, y, r) = (x.buffer(), y.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        distinct(r, "result", y, "y")?;
        // SAFETY: as `axpy`; `result` has at least one element and is distinct from the inputs.
        let raw = unsafe {
            T::dotc_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0] = ||x||_2`.
    ///
    /// # Errors
    ///
    /// As [`Scope::dot`].
    pub fn nrm2<T: BlasElement>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        result: &Write<'scope, T::Real>,
    ) -> Result<Event<'scope>> {
        let (x, r) = (x.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        // SAFETY: as `dot`.
        let raw = unsafe {
            T::nrm2_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0] = sum |x_i|`.
    ///
    /// # Errors
    ///
    /// As [`Scope::dot`].
    pub fn asum<T: BlasElement>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        result: &Write<'scope, T::Real>,
    ) -> Result<Event<'scope>> {
        let (x, r) = (x.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        // SAFETY: as `dot`.
        let raw = unsafe {
            T::asum_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0]` = zero-based index of the first element of largest magnitude.
    ///
    /// # Errors
    ///
    /// As [`Scope::dot`].
    pub fn iamax<T: BlasElement>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        result: &Write<'scope, i64>,
    ) -> Result<Event<'scope>> {
        let (x, r) = (x.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        // SAFETY: as `dot`.
        let raw = unsafe {
            T::iamax_raw_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `y = alpha * op(A) * x + beta * y`, `A` is `m x n` column-major.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `y` is the same buffer as `a` or
    /// `x`; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemv<T: BlasElement>(
        &'scope self,
        trans: Transpose,
        m: usize,
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
        let (x_len, y_len) = if trans == Transpose::No {
            (n, m)
        } else {
            (m, n)
        };
        check_matrix("A", a.len(), lda, m, n)?;
        check_vec("x", x.len(), x_len, incx)?;
        check_vec("y", y.len(), y_len, incy)?;
        distinct(y, "y", a, "A")?;
        distinct(y, "y", x, "x")?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::gemv_raw_async(
                self.queue.raw(),
                trans.code(),
                dim("m", m)?,
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
}
