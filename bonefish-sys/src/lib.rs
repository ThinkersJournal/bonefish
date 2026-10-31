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
        /// 1 if the queue is in-order (every queue from `bf_queue_new` is), 0 if not, -1 on failure.
        pub fn bf_queue_is_in_order(q: *mut c_void) -> i32;
        /// Waits for all work on the queue. Returns 1 (message in `bf_last_error`) if that surfaced
        /// an asynchronous error.
        pub fn bf_queue_wait(q: *mut c_void) -> i32;
        /// Waits for one event (`wait_and_throw`); same status convention as `bf_queue_wait`.
        pub fn bf_event_wait(e: *mut c_void) -> i32;
        pub fn bf_event_free(e: *mut c_void);
        /// Diagnostic: submits a host task that throws; the failure is asynchronous. Returns the
        /// event, or null on failure.
        pub fn bf_test_async_failure(q: *mut c_void) -> *mut c_void;
        /// Non-blocking `gemm`: returns the event (null on failure). The device may still be
        /// reading and writing the operands when this returns; the caller keeps them alive and
        /// untouched until the event or the queue has been waited on. `deps` points at `n_deps`
        /// event handles to wait for (null with 0 for none).
        #[allow(clippy::too_many_arguments)]
        pub fn bf_sgemm_async(
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
            deps: *const *mut c_void,
            n_deps: i64,
        ) -> *mut c_void;
        #[allow(clippy::too_many_arguments)]
        pub fn bf_dgemm_async(
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
            deps: *const *mut c_void,
            n_deps: i64,
        ) -> *mut c_void;
        /// Diagnostic: how many oneMKL calls have entered the shim's lock path. A test uses it to
        /// prove no oneMKL-calling function bypasses the lock. `-1` in a build without the shim.
        pub fn bf_mkl_lock_acquisitions() -> i64;
        /// Message of the last failure on this thread.
        pub fn bf_last_error() -> *const c_char;
    }
}

#[cfg(bonefish_sycl)]
pub use ffi::*;

/// Whether the shim was built into this crate.
pub const SYCL_AVAILABLE: bool = cfg!(bonefish_sycl);

/// Stand-ins used when the shim was not built (no oneAPI at build time): same signatures as the
/// real foreign functions, so everything above this crate compiles identically either way.
/// Constructors return null, status-returning calls return [`STATUS_UNAVAILABLE`], and
/// `bf_last_error` explains. The real/stub pair is checked by the compiler in the two CI
/// configurations (with and without oneAPI), because the safe layer calls whichever is active.
// The stand-ins must be `unsafe` to match the real foreign declarations (callers wrap them in
// `unsafe` either way), and they take the same long argument lists, so these two lints fire on
// them for reasons that do not apply: they dereference nothing and have no safety contract.
#[cfg(not(bonefish_sycl))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
mod stub {
    use std::ffi::{c_char, c_void};

    /// Returned by every status-returning call in a build without the shim.
    pub const STATUS_UNAVAILABLE: i32 = 4;
    const UNAVAILABLE_MSG: &[u8] =
        b"bonefish was built without oneAPI (icpx not found at build time)\0";

    pub unsafe extern "C" fn bf_queue_new(_kind: i32) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub unsafe extern "C" fn bf_queue_free(_q: *mut c_void) {}

    pub unsafe extern "C" fn bf_queue_device_name(
        _q: *mut c_void,
        _buf: *mut c_char,
        _cap: i64,
    ) -> i64 {
        -1
    }

    pub unsafe extern "C" fn bf_malloc(_q: *mut c_void, _kind: i32, _bytes: i64) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub unsafe extern "C" fn bf_free(_q: *mut c_void, _p: *mut c_void) {}

    pub unsafe extern "C" fn bf_memcpy(
        _q: *mut c_void,
        _dst: *mut c_void,
        _src: *const c_void,
        _bytes: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_sgemm(
        _q: *mut c_void,
        _ta: i32,
        _tb: i32,
        _m: i64,
        _n: i64,
        _k: i64,
        _alpha: f32,
        _a: *const f32,
        _lda: i64,
        _b: *const f32,
        _ldb: i64,
        _beta: f32,
        _c: *mut f32,
        _ldc: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dgemm(
        _q: *mut c_void,
        _ta: i32,
        _tb: i32,
        _m: i64,
        _n: i64,
        _k: i64,
        _alpha: f64,
        _a: *const f64,
        _lda: i64,
        _b: *const f64,
        _ldb: i64,
        _beta: f64,
        _c: *mut f64,
        _ldc: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dft_c2c_new(
        _q: *mut c_void,
        _prec: i32,
        _ndim: i64,
        _lengths: *const i64,
        _inplace: i32,
        _fwd_scale: f64,
        _bwd_scale: f64,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub unsafe extern "C" fn bf_dft_free(_h: *mut c_void) {}

    pub unsafe extern "C" fn bf_dft_compute(
        _h: *mut c_void,
        _dir: i32,
        _input: *mut c_void,
        _out: *mut c_void,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_saxpy(
        _q: *mut c_void,
        _n: i64,
        _alpha: f32,
        _x: *const f32,
        _incx: i64,
        _y: *mut f32,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_sscal(
        _q: *mut c_void,
        _n: i64,
        _alpha: f32,
        _x: *mut f32,
        _incx: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_scopy(
        _q: *mut c_void,
        _n: i64,
        _x: *const f32,
        _incx: i64,
        _y: *mut f32,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_sswap(
        _q: *mut c_void,
        _n: i64,
        _x: *mut f32,
        _incx: i64,
        _y: *mut f32,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_sdot(
        _q: *mut c_void,
        _n: i64,
        _x: *const f32,
        _incx: i64,
        _y: *const f32,
        _incy: i64,
        _result: *mut f32,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_snrm2(
        _q: *mut c_void,
        _n: i64,
        _x: *const f32,
        _incx: i64,
        _result: *mut f32,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_sasum(
        _q: *mut c_void,
        _n: i64,
        _x: *const f32,
        _incx: i64,
        _result: *mut f32,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_siamax(
        _q: *mut c_void,
        _n: i64,
        _x: *const f32,
        _incx: i64,
        _result: *mut i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_sgemv(
        _q: *mut c_void,
        _trans: i32,
        _m: i64,
        _n: i64,
        _alpha: f32,
        _a: *const f32,
        _lda: i64,
        _x: *const f32,
        _incx: i64,
        _beta: f32,
        _y: *mut f32,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_daxpy(
        _q: *mut c_void,
        _n: i64,
        _alpha: f64,
        _x: *const f64,
        _incx: i64,
        _y: *mut f64,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dscal(
        _q: *mut c_void,
        _n: i64,
        _alpha: f64,
        _x: *mut f64,
        _incx: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dcopy(
        _q: *mut c_void,
        _n: i64,
        _x: *const f64,
        _incx: i64,
        _y: *mut f64,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dswap(
        _q: *mut c_void,
        _n: i64,
        _x: *mut f64,
        _incx: i64,
        _y: *mut f64,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_ddot(
        _q: *mut c_void,
        _n: i64,
        _x: *const f64,
        _incx: i64,
        _y: *const f64,
        _incy: i64,
        _result: *mut f64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dnrm2(
        _q: *mut c_void,
        _n: i64,
        _x: *const f64,
        _incx: i64,
        _result: *mut f64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dasum(
        _q: *mut c_void,
        _n: i64,
        _x: *const f64,
        _incx: i64,
        _result: *mut f64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_diamax(
        _q: *mut c_void,
        _n: i64,
        _x: *const f64,
        _incx: i64,
        _result: *mut i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_dgemv(
        _q: *mut c_void,
        _trans: i32,
        _m: i64,
        _n: i64,
        _alpha: f64,
        _a: *const f64,
        _lda: i64,
        _x: *const f64,
        _incx: i64,
        _beta: f64,
        _y: *mut f64,
        _incy: i64,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_queue_is_in_order(_q: *mut c_void) -> i32 {
        -1
    }

    pub unsafe extern "C" fn bf_queue_wait(_q: *mut c_void) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_event_wait(_e: *mut c_void) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub unsafe extern "C" fn bf_event_free(_e: *mut c_void) {}

    pub unsafe extern "C" fn bf_test_async_failure(_q: *mut c_void) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub unsafe extern "C" fn bf_sgemm_async(
        _q: *mut c_void,
        _ta: i32,
        _tb: i32,
        _m: i64,
        _n: i64,
        _k: i64,
        _alpha: f32,
        _a: *const f32,
        _lda: i64,
        _b: *const f32,
        _ldb: i64,
        _beta: f32,
        _c: *mut f32,
        _ldc: i64,
        _deps: *const *mut c_void,
        _n_deps: i64,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub unsafe extern "C" fn bf_dgemm_async(
        _q: *mut c_void,
        _ta: i32,
        _tb: i32,
        _m: i64,
        _n: i64,
        _k: i64,
        _alpha: f64,
        _a: *const f64,
        _lda: i64,
        _b: *const f64,
        _ldb: i64,
        _beta: f64,
        _c: *mut f64,
        _ldc: i64,
        _deps: *const *mut c_void,
        _n_deps: i64,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub unsafe extern "C" fn bf_mkl_lock_acquisitions() -> i64 {
        -1
    }

    pub unsafe extern "C" fn bf_last_error() -> *const c_char {
        UNAVAILABLE_MSG.as_ptr().cast()
    }
}

#[cfg(not(bonefish_sycl))]
pub use stub::*;
