//! One-time, per-process warm-up of the oneMKL BLAS routines (ROADMAP R12).
//!
//! On Windows (oneAPI 2026.1, CPU device) running *different* oneMKL routines for the first time on
//! parallel threads crashed about 1 run in 10: a delay-load failure inside `mkl_sycl_blas.6.dll`
//! (it loads `mkl_tbb_thread.3.dll!mkl_serv_verbose_mode`), then a jump to an address in no module.
//! Running every routine once, serially, first gave 0 crashes in 100 on the same binary. The
//! **mechanism is not shown** and this is a guard, not a fix: R12 stays open.
//!
//! The first queue created in the process runs one tiny call of each routine family, once; any
//! other thread creating a queue waits for it (`Once`). Failures are ignored on purpose: warming
//! up must never turn a working queue into an error, and the real call reports its own errors.
//! Set `BONEFISH_NO_WARMUP=1` to skip it.

use std::sync::Once;

use crate::blas;
use crate::scalar::{BlasElement, GemmElement};
use crate::{Queue, Transpose, UsmBuffer, UsmKind};

static WARMED: Once = Once::new();

/// Warm every routine family once, if `queue` is the first one in this process.
pub(crate) fn once(queue: &Queue) {
    if !crate::SYCL_AVAILABLE || std::env::var_os("BONEFISH_NO_WARMUP").is_some_and(|v| v == "1") {
        return;
    }
    WARMED.call_once(|| {
        use crate::Complex;
        warm::<f32>(queue, 1.0);
        warm::<f64>(queue, 1.0);
        warm::<Complex<f32>>(queue, Complex::new(1.0, 0.0));
        warm::<Complex<f64>>(queue, Complex::new(1.0, 0.0));
    });
}

fn warm<T: BlasElement + GemmElement + Copy>(q: &Queue, one: T) {
    let buf = |kind| UsmBuffer::from_slice(q, kind, &[one; 4]);
    let (Ok(x), Ok(mut y), Ok(mut z)) = (
        buf(UsmKind::Shared),
        buf(UsmKind::Shared),
        buf(UsmKind::Shared),
    ) else {
        return;
    };
    let _ = blas::axpy(q, 4, one, &x, 1, &mut y, 1);
    let _ = blas::scal(q, 4, one, &mut y, 1);
    let _ = blas::copy(q, 4, &x, 1, &mut y, 1);
    let _ = blas::swap(q, 4, &mut z, 1, &mut y, 1);
    let _ = blas::dot(q, 4, &x, 1, &y, 1);
    let _ = blas::nrm2(q, 4, &x, 1);
    let _ = blas::asum(q, 4, &x, 1);
    let _ = blas::iamax(q, 4, &x, 1);
    let _ = blas::gemv(q, Transpose::No, 2, 2, one, &x, 2, &z, 1, one, &mut y, 1);
    let _ = blas::gemm(
        q,
        Transpose::No,
        Transpose::No,
        2,
        2,
        2,
        one,
        &x,
        2,
        &z,
        2,
        one,
        &mut y,
        2,
    );
}
