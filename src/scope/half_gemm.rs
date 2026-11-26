//! Half-precision `gemm` on [`Scope`] (non-blocking): `half::f16` and `half::bf16` inputs, with the
//! oneMKL variant named in the method. Extents and output aliasing are checked in Rust.
//!
//! Variants (oneMKL `usm_decls.hpp`, as `(A, B, C, scalar)`):
//!
//! | method | A, B | C | accumulation |
//! |---|---|---|---|
//! | [`Scope::hgemm`] | `f16` | `f16` | unstated by oneMKL |
//! | [`Scope::hgemm_f32`] | `f16` | `f32` | f32 scalars and output |
//! | [`Scope::bgemm`] | `bf16` | `bf16` | f32 scalars |
//! | [`Scope::bgemm_f32`] | `bf16` | `f32` | f32 scalars and output |
//!
//! `alpha` and `beta` are `f32` in every variant. Whether a given device runs a variant is not
//! assumed: an unsupported one returns the shim's error, never a wrong answer.

use std::ffi::c_void;

use bonefish_sys as sys;
use half::{bf16, f16};

use super::{Event, Operand, Scope, Write};
use crate::error::{Error, Result};
use crate::layout::{Transpose, check_matrix, dim};

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
    *const u16,
    i64,
    f32,
    *mut P,
    i64,
    *const *mut c_void,
    i64,
) -> *mut c_void;

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    #[allow(clippy::too_many_arguments)]
    fn half_gemm<A: crate::scalar::Element, C: crate::scalar::Element, P>(
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
        b: &impl Operand<A>,
        ldb: usize,
        beta: f32,
        c: &Write<'scope, C>,
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
        if std::ptr::eq(a.raw().cast::<u8>(), c.raw().cast::<u8>())
            || std::ptr::eq(b.raw().cast::<u8>(), c.raw().cast::<u8>())
        {
            return Err(Error::InvalidArgument(
                "the output buffer is also an input of the same operation".into(),
            ));
        }
        // SAFETY: extents were checked above against each buffer; USM of this scope's queue,
        // borrowed until the scope ends, which waits for the queue first. `f16` and `bf16` are
        // `repr(transparent)` over `u16` (the `half` crate's documented layout), the 16-bit word
        // the shim takes; `c` is distinct from `a` and `b`; the shim chains this submission to the
        // queue's previous one.
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
                b.raw().cast(),
                dim("ldb", ldb)?,
                beta,
                c.raw().cast(),
                dim("ldc", ldc)?,
                std::ptr::null(),
                0,
            )
        };
        self.event(raw)
    }
}

macro_rules! half_method {
    ($(#[$doc:meta])* $name:ident, $a:ty, $c:ty, $p:ty, $shim:ident) => {
        impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
            $(#[$doc])*
            ///
            /// Non-blocking, column-major `c = alpha * op(a) * op(b) + beta * c`.
            ///
            /// # Errors
            ///
            /// [`Error::InvalidArgument`] for an extent that does not fit its buffer or when `c`
            /// is the same buffer as `a` or `b`; [`Error::Sycl`]/[`Error::Mkl`]/[`Error::Other`]
            /// if submission fails, including a device that lacks this variant.
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
                b: &impl Operand<$a>,
                ldb: usize,
                beta: f32,
                c: &Write<'scope, $c>,
                ldc: usize,
            ) -> Result<Event<'scope>> {
                self.half_gemm::<$a, $c, $p>(
                    sys::$shim, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                )
            }
        }
    };
}

half_method!(
    /// `f16` inputs, `f16` output (oneMKL's all-half variant; its internal accumulation is unstated).
    hgemm,
    f16,
    f16,
    u16,
    bf_hgemm_async
);
half_method!(
    /// `f16` inputs, `f32` output and scalars.
    hgemm_f32,
    f16,
    f32,
    f32,
    bf_hgemm_f32_async
);
half_method!(
    /// `bf16` inputs, `bf16` output, `f32` scalars.
    bgemm,
    bf16,
    bf16,
    u16,
    bf_bgemm_async
);
half_method!(
    /// `bf16` inputs, `f32` output and scalars.
    bgemm_f32,
    bf16,
    f32,
    f32,
    bf_bgemm_f32_async
);
