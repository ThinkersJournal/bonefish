//! Strided batched `gemm` on [`Scope`] (non-blocking) for `f32` and `f64`: `batch` independent
//! products, operand `i` at `base + i * stride` elements. Extents, strides and output aliasing are
//! checked in Rust. The blocking form is [`crate::blas::gemm_batch_strided`].

use std::ffi::c_void;

use bonefish_sys as sys;

use super::{Event, Operand, Scope, Write};
use crate::blas::{Transpose, check_matrix, dim};
use crate::error::{Error, Result};
use crate::scalar::Scalar;

/// Element types with a strided batched `gemm`: `f32` and `f64`. Sealed through [`Element`].
pub trait BatchedGemm: Scalar {
    /// # Safety
    /// Pointers are USM from `q`; operand `i` of `batch` lies at `base + i * stride` and covers
    /// the matrix extent; alive and untouched by the host until the event (or queue) is waited
    /// on; the `c` matrices do not overlap each other or the inputs.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_batch_strided_raw_async(
        q: *mut c_void,
        ta: i32,
        tb: i32,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        stride_a: i64,
        b: *const Self,
        ldb: i64,
        stride_b: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
        stride_c: i64,
        batch: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}

macro_rules! impl_batched {
    ($t:ty, $f:ident) => {
        impl BatchedGemm for $t {
            unsafe fn gemm_batch_strided_raw_async(
                q: *mut c_void,
                ta: i32,
                tb: i32,
                m: i64,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                stride_a: i64,
                b: *const Self,
                ldb: i64,
                stride_b: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                stride_c: i64,
                batch: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: forwarded from the caller's contract on the trait method.
                unsafe {
                    sys::$f(
                        q, ta, tb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c,
                        ldc, stride_c, batch, deps, n_deps,
                    )
                }
            }
        }
    };
}

impl_batched!(f32, bf_sgemm_batch_strided_async);
impl_batched!(f64, bf_dgemm_batch_strided_async);

/// Check `batch` column-major `rows x cols` matrices, `ld` apart in rows, `stride` elements apart,
/// fit in `len`. `distinct` additionally requires the matrices not to overlap (outputs).
pub(crate) fn check_batched(
    name: &str,
    len: usize,
    ld: usize,
    stride: usize,
    rows: usize,
    cols: usize,
    batch: usize,
    distinct: bool,
) -> Result<()> {
    check_matrix(name, len, ld, rows, cols)?;
    if batch == 0 || rows == 0 || cols == 0 {
        return Ok(());
    }
    let one = (cols - 1) * ld + rows; // check_matrix proved this does not overflow
    if distinct && batch > 1 && stride < one {
        return Err(Error::InvalidArgument(format!(
            "{name}: stride {stride} is below one matrix ({one} elements), so the batch overlaps"
        )));
    }
    let needed = (batch - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(one))
        .ok_or_else(|| Error::InvalidArgument(format!("{name}: batch extent overflows")))?;
    if needed > len {
        return Err(Error::InvalidArgument(format!(
            "{name}: {batch} matrices at stride {stride} need {needed} elements, buffer has {len}"
        )));
    }
    Ok(())
}

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// Non-blocking strided batched `c_i = alpha * op(a_i) * op(b_i) + beta * c_i` for
    /// `i < batch`, column-major, `a_i` at `a + i * stride_a` elements and likewise `b`, `c`. A
    /// stride of 0 on `a` or `b` reuses one matrix for the whole batch; `stride_c` must be at
    /// least one `c` matrix so outputs do not overlap.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent (including the whole batch) does not fit its
    /// buffer, `stride_c` is too small, or `c` is the same buffer as `a` or `b`; submission
    /// errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batch_strided<T: BatchedGemm>(
        &'scope self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        stride_a: usize,
        b: &impl Operand<T>,
        ldb: usize,
        stride_b: usize,
        beta: T,
        c: &Write<'scope, T>,
        ldc: usize,
        stride_c: usize,
        batch: usize,
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
        check_batched("A", a.len(), lda, stride_a, a_rows, a_cols, batch, false)?;
        check_batched("B", b.len(), ldb, stride_b, b_rows, b_cols, batch, false)?;
        check_batched("C", c.len(), ldc, stride_c, m, n, batch, true)?;
        if std::ptr::eq(a.raw(), c.raw()) || std::ptr::eq(b.raw(), c.raw()) {
            return Err(Error::InvalidArgument(
                "the output buffer is also an input of the same operation".into(),
            ));
        }
        // SAFETY: extents of the whole batch were checked above against each buffer; USM of this
        // scope's queue, borrowed until the scope ends, which waits for the queue first. `c` is
        // distinct from `a` and `b`, its matrices do not overlap (stride_c checked), and the shim
        // chains this submission to the queue's previous one.
        let raw = unsafe {
            T::gemm_batch_strided_raw_async(
                self.queue.raw(),
                transa.code(),
                transb.code(),
                dim("m", m)?,
                dim("n", n)?,
                dim("k", k)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                dim("stride_a", stride_a)?,
                b.raw(),
                dim("ldb", ldb)?,
                dim("stride_b", stride_b)?,
                beta,
                c.raw(),
                dim("ldc", ldc)?,
                dim("stride_c", stride_c)?,
                dim("batch", batch)?,
                std::ptr::null(),
                0,
            )
        };
        self.event(raw)
    }
}
