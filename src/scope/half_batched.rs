//! Strided batched half-precision `gemm` on [`Scope`] (non-blocking): `half::f16` and `half::bf16`
//! inputs, the four oneMKL `gemm_batch` variants named as in [`super::half_gemm`]. Operand `i` is
//! at `base + i * stride` elements; extents of the whole batch, `stride_c` and output aliasing are
//! checked in Rust. `alpha` and `beta` are `f32`. The blocking forms are in [`crate::blas`].

use std::ffi::c_void;

use bonefish_sys as sys;
use half::{bf16, f16};

use super::batched::check_batched;
use super::{Event, Operand, Scope, Write};
use crate::blas::{Transpose, dim};
use crate::error::{Error, Result};

/// The shared signature of the four shim functions: 16-bit words for `A` and `B`.
type Shim<P> = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i64,
    i64,
    i64,
    f32,
    *const u16,
    i64,
    i64,
    *const u16,
    i64,
    i64,
    f32,
    *mut P,
    i64,
    i64,
    i64,
    *const *mut c_void,
    i64,
) -> *mut c_void;

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    #[allow(clippy::too_many_arguments)]
    fn half_gemm_batch<A: crate::scalar::Element, C: crate::scalar::Element, P>(
        &'scope self,
        shim: Shim<P>,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &impl Operand<A>,
        lda: usize,
        stride_a: usize,
        b: &impl Operand<A>,
        ldb: usize,
        stride_b: usize,
        beta: f32,
        c: &Write<'scope, C>,
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
        if std::ptr::eq(a.raw().cast::<u8>(), c.raw().cast::<u8>())
            || std::ptr::eq(b.raw().cast::<u8>(), c.raw().cast::<u8>())
        {
            return Err(Error::InvalidArgument(
                "the output buffer is also an input of the same operation".into(),
            ));
        }
        // SAFETY: extents of the whole batch were checked above against each buffer; USM of this
        // scope's queue, borrowed until the scope ends, which waits for the queue first. `f16` and
        // `bf16` are `repr(transparent)` over `u16` (the `half` crate's documented layout), the
        // 16-bit word the shim takes; `c` is distinct from `a` and `b`, its matrices do not overlap
        // (stride_c checked), and the shim chains this submission to the queue's previous one.
        let raw = unsafe {
            shim(
                self.queue.raw(),
                transa.code(),
                transb.code(),
                dim("m", m)?,
                dim("n", n)?,
                dim("k", k)?,
                alpha,
                a.raw().cast(),
                dim("lda", lda)?,
                dim("stride_a", stride_a)?,
                b.raw().cast(),
                dim("ldb", ldb)?,
                dim("stride_b", stride_b)?,
                beta,
                c.raw().cast(),
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

macro_rules! half_batch_method {
    ($(#[$doc:meta])* $name:ident, $a:ty, $c:ty, $p:ty, $shim:ident) => {
        impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
            $(#[$doc])*
            ///
            /// Non-blocking, column-major strided batched
            /// `c_i = alpha * op(a_i) * op(b_i) + beta * c_i` for `i < batch`, `a_i` at
            /// `a + i * stride_a` elements and likewise `b`, `c`; a stride of 0 on `a` or `b`
            /// reuses one matrix; `stride_c` must be at least one `c` matrix.
            ///
            /// # Errors
            ///
            /// [`Error::InvalidArgument`] if an extent (including the whole batch) does not fit
            /// its buffer, `stride_c` is too small, or `c` is the same buffer as `a` or `b`;
            /// [`Error::Sycl`]/[`Error::Mkl`]/[`Error::Other`] if submission fails, including a
            /// device that lacks this variant.
            #[allow(clippy::too_many_arguments)]
            pub fn $name(
                &'scope self,
                transa: Transpose,
                transb: Transpose,
                m: usize,
                n: usize,
                k: usize,
                alpha: f32,
                a: &impl Operand<$a>,
                lda: usize,
                stride_a: usize,
                b: &impl Operand<$a>,
                ldb: usize,
                stride_b: usize,
                beta: f32,
                c: &Write<'scope, $c>,
                ldc: usize,
                stride_c: usize,
                batch: usize,
            ) -> Result<Event<'scope>> {
                self.half_gemm_batch::<$a, $c, $p>(
                    sys::$shim, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                    stride_b, beta, c, ldc, stride_c, batch,
                )
            }
        }
    };
}

half_batch_method!(
    /// `f16` inputs, `f16` output (oneMKL's all-half variant; its internal accumulation is
    /// unstated).
    hgemm_batch_strided,
    f16,
    f16,
    u16,
    bf_hgemm_batch_strided_async
);
half_batch_method!(
    /// `f16` inputs, `f32` output and scalars.
    hgemm_f32_batch_strided,
    f16,
    f32,
    f32,
    bf_hgemm_f32_batch_strided_async
);
half_batch_method!(
    /// `bf16` inputs, `bf16` output, `f32` scalars.
    bgemm_batch_strided,
    bf16,
    bf16,
    u16,
    bf_bgemm_batch_strided_async
);
half_batch_method!(
    /// `bf16` inputs, `f32` output and scalars.
    bgemm_f32_batch_strided,
    bf16,
    f32,
    f32,
    bf_bgemm_f32_batch_strided_async
);
