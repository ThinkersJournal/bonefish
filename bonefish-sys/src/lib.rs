//! Raw FFI to Bonefish's C shim over oneMKL's SYCL interface. See the `bonefish` crate for the
//! safe API.
//!
//! The shim is only built when Intel's DPC++ compiler is found at build time; otherwise this
//! crate is empty and `cfg(bonefish_sycl)` is not set. Handles are opaque pointers; status codes
//! are `0` ok, `1` SYCL error, `2` oneMKL error, `3` other, with the message from
//! [`bf_last_error`].

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
        /// Message of the last failure on this thread.
        pub fn bf_last_error() -> *const c_char;
    }
}

#[cfg(bonefish_sycl)]
pub use ffi::*;

/// Whether the shim was built into this crate.
pub const SYCL_AVAILABLE: bool = cfg!(bonefish_sycl);
