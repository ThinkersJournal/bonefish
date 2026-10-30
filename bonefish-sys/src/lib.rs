//! Raw FFI to Bonefish's C shim over oneMKL's SYCL interface. See the `bonefish` crate for the
//! safe API.
//!
//! The shim is only built when Intel's DPC++ compiler is found at build time; otherwise this
//! crate is empty and `cfg(bonefish_sycl)` is not set. Handles are opaque pointers; status codes
//! are `0` ok, `1` SYCL error, `2` oneMKL error, `3` other, with the message from
//! `bf_last_error`.

#[cfg(bonefish_sycl)]
mod ffi {
    use std::ffi::{c_char, c_void};

    unsafe extern "C" {
        /// `kind`: 0 default selector, 1 CPU, 2 GPU. Null on failure.
        pub fn bf_queue_new(kind: i32) -> *mut c_void;
        pub fn bf_queue_free(q: *mut c_void);
        /// Copies the NUL-terminated device name into `buf` (truncated to `cap`); returns the
        /// full length, or -1 on failure.
        pub fn bf_queue_device_name(q: *mut c_void, buf: *mut c_char, cap: i64) -> i64;
        /// `kind`: 0 shared, 1 device, 2 host USM. Null on failure.
        pub fn bf_malloc(q: *mut c_void, kind: i32, bytes: i64) -> *mut c_void;
        pub fn bf_free(q: *mut c_void, p: *mut c_void);
        pub fn bf_memcpy(q: *mut c_void, dst: *mut c_void, src: *const c_void, bytes: i64) -> i32;
        /// Column-major, blocking. `ta`/`tb`: 0 N, 1 T, 2 C.
        #[allow(clippy::too_many_arguments)]
        pub fn bf_sgemm(
            q: *mut c_void,
            ta: i32,
            tb: i32,
            m: i64,
            n: i64,
            k: i64,
            alpha: f32,
            a: *const f32,
            lda: i64,
            b: *const f32,
            ldb: i64,
            beta: f32,
            c: *mut f32,
            ldc: i64,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn bf_dgemm(
            q: *mut c_void,
            ta: i32,
            tb: i32,
            m: i64,
            n: i64,
            k: i64,
            alpha: f64,
            a: *const f64,
            lda: i64,
            b: *const f64,
            ldb: i64,
            beta: f64,
            c: *mut f64,
            ldc: i64,
        ) -> i32;
        /// Committed complex-to-complex descriptor over `ndim` `lengths`. `prec`: 0 single, 1 double.
        /// `fwd_scale`/`bwd_scale` multiply the respective result (oneMKL defaults: 1 and 1).
        /// Null on failure.
        pub fn bf_dft_c2c_new(
            q: *mut c_void,
            prec: i32,
            ndim: i64,
            lengths: *const i64,
            inplace: i32,
            fwd_scale: f64,
            bwd_scale: f64,
        ) -> *mut c_void;
        pub fn bf_dft_free(h: *mut c_void);
        /// Blocking transform on USM interleaved complex data. `dir`: 0 forward, 1 backward. Pass
        /// `out` null for an in-place descriptor (`input` is overwritten).
        pub fn bf_dft_compute(
            h: *mut c_void,
            dir: i32,
            input: *mut c_void,
            out: *mut c_void,
        ) -> i32;
        pub fn bf_saxpy(
            q: *mut c_void,
            n: i64,
            alpha: f32,
            x: *const f32,
            incx: i64,
            y: *mut f32,
            incy: i64,
        ) -> i32;
        pub fn bf_sscal(q: *mut c_void, n: i64, alpha: f32, x: *mut f32, incx: i64) -> i32;
        pub fn bf_scopy(
            q: *mut c_void,
            n: i64,
            x: *const f32,
            incx: i64,
            y: *mut f32,
            incy: i64,
        ) -> i32;
        pub fn bf_sswap(
            q: *mut c_void,
            n: i64,
            x: *mut f32,
            incx: i64,
            y: *mut f32,
            incy: i64,
        ) -> i32;
        /// Scalar results (`dot`, `nrm2`, `asum`, `iamax`) are written through a USM `result`.
        pub fn bf_sdot(
            q: *mut c_void,
            n: i64,
            x: *const f32,
            incx: i64,
            y: *const f32,
            incy: i64,
            result: *mut f32,
        ) -> i32;
        pub fn bf_snrm2(q: *mut c_void, n: i64, x: *const f32, incx: i64, result: *mut f32) -> i32;
        pub fn bf_sasum(q: *mut c_void, n: i64, x: *const f32, incx: i64, result: *mut f32) -> i32;
        /// Zero-based index of the first element of largest magnitude.
        pub fn bf_siamax(q: *mut c_void, n: i64, x: *const f32, incx: i64, result: *mut i64)
        -> i32;
        /// `y = alpha * op(A) * x + beta * y`, `A` is `m x n` column-major. `trans`: 0 N, 1 T, 2 C.
        pub fn bf_sgemv(
            q: *mut c_void,
            trans: i32,
            m: i64,
            n: i64,
            alpha: f32,
            a: *const f32,
            lda: i64,
            x: *const f32,
            incx: i64,
            beta: f32,
            y: *mut f32,
            incy: i64,
        ) -> i32;
        pub fn bf_daxpy(
            q: *mut c_void,
            n: i64,
            alpha: f64,
            x: *const f64,
            incx: i64,
            y: *mut f64,
            incy: i64,
        ) -> i32;
        pub fn bf_dscal(q: *mut c_void, n: i64, alpha: f64, x: *mut f64, incx: i64) -> i32;
        pub fn bf_dcopy(
            q: *mut c_void,
            n: i64,
            x: *const f64,
            incx: i64,
            y: *mut f64,
            incy: i64,
        ) -> i32;
        pub fn bf_dswap(
            q: *mut c_void,
            n: i64,
            x: *mut f64,
            incx: i64,
            y: *mut f64,
            incy: i64,
        ) -> i32;
        /// Scalar results (`dot`, `nrm2`, `asum`, `iamax`) are written through a USM `result`.
        pub fn bf_ddot(
            q: *mut c_void,
            n: i64,
            x: *const f64,
            incx: i64,
            y: *const f64,
            incy: i64,
            result: *mut f64,
        ) -> i32;
        pub fn bf_dnrm2(q: *mut c_void, n: i64, x: *const f64, incx: i64, result: *mut f64) -> i32;
        pub fn bf_dasum(q: *mut c_void, n: i64, x: *const f64, incx: i64, result: *mut f64) -> i32;
        /// Zero-based index of the first element of largest magnitude.
        pub fn bf_diamax(q: *mut c_void, n: i64, x: *const f64, incx: i64, result: *mut i64)
        -> i32;
        /// `y = alpha * op(A) * x + beta * y`, `A` is `m x n` column-major. `trans`: 0 N, 1 T, 2 C.
        pub fn bf_dgemv(
            q: *mut c_void,
            trans: i32,
            m: i64,
            n: i64,
            alpha: f64,
            a: *const f64,
            lda: i64,
            x: *const f64,
            incx: i64,
            beta: f64,
            y: *mut f64,
            incy: i64,
        ) -> i32;
        /// Message of the last failure on this thread.
        pub fn bf_last_error() -> *const c_char;
    }
}

#[cfg(bonefish_sycl)]
pub use ffi::*;

/// Whether the shim was built into this crate.
pub const SYCL_AVAILABLE: bool = cfg!(bonefish_sycl);
