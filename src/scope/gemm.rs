//! Non-blocking `gemm` on [`Scope`]: [`Scope::gemm`] and [`Scope::gemm_after`] (moved out of
//! `scope.rs`, R15). The other level 3 operations are in `level3.rs`, half precision in
//! `half_gemm.rs`, strided batches in `batched.rs`.

use std::ffi::c_void;

use super::{Event, Operand, Scope, Write};
use crate::error::{Error, Result};
use crate::layout::{Transpose, check_matrix, dim};
use crate::scalar::GemmElement;

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// Non-blocking `c = alpha * op(a) * op(b) + beta * c`, column-major. Returns once the work is
    /// submitted; the device finishes it later, at the latest when the scope ends.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an extent that does not fit its buffer, or when `c` is the
    /// same buffer as `a` or `b`; [`Error::Sycl`]/[`Error::Mkl`]/[`Error::Other`] if submission
    /// fails.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<T: GemmElement>(
        &'scope self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
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
        self.gemm_after(
            &[],
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
    }

    /// As [`Scope::gemm`], but the work also waits for every event in `after`. Earlier work in
    /// the scope is already ordered before this (the shim chains submissions); this is for stating
    /// a dependency explicitly.
    ///
    /// # Errors
    ///
    /// As [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_after<T: GemmElement>(
        &'scope self,
        after: &[&Event<'scope>],
        transa: Transpose,
        transb: Transpose,
        m: usize,
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
        let (a_rows, a_cols) = if transa == Transpose::No {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if transb == Transpose::No {
            (k, n)
        } else {
            (n, k)
        };
        check_matrix("A", a.len(), lda, a_rows, a_cols)?;
        check_matrix("B", b.len(), ldb, b_rows, b_cols)?;
        check_matrix("C", c.len(), ldc, m, n)?;
        if std::ptr::eq(a.raw(), c.raw()) || std::ptr::eq(b.raw(), c.raw()) {
            return Err(Error::InvalidArgument(
                "the output buffer is also an input of the same operation".into(),
            ));
        }
        let deps: Vec<*mut c_void> = after.iter().map(|e| e.raw.as_ptr()).collect();
        // SAFETY: every pointer is USM from this scope's queue (checked when bound) and the
        // extents above fit inside the buffers. The buffers stay borrowed until the scope ends,
        // and the scope waits for the queue before that, so the device is finished with them
        // first. `c` is distinct from `a` and `b` (checked above), and the shim chains this
        // submission to the queue's previous one, so it cannot overlap earlier ones that name the
        // same buffers. `deps` are live events of this scope.
        let raw = unsafe {
            T::gemm_raw_async(
                self.queue.raw(),
                transa.code(),
                transb.code(),
                dim("m", m)?,
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
                if deps.is_empty() {
                    std::ptr::null()
                } else {
                    deps.as_ptr()
                },
                deps.len() as i64,
            )
        };
        self.event(raw)
    }
}
