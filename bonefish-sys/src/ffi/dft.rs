// SPDX-License-Identifier: MIT OR Apache-2.0
//! Foreign declarations: complex and real DFT. Split out of `lib.rs` (R15).

use std::ffi::c_void;

unsafe extern "C" {
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
    pub fn bf_dft_compute(h: *mut c_void, dir: i32, input: *mut c_void, out: *mut c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dft_compute_async(
        h: *mut c_void,
        dir: i32,
        input: *mut c_void,
        out: *mut c_void,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    /// Committed real DFT descriptor (out-of-place only) over `ndim` `lengths`. `prec`: 0
    /// single, 1 double. Null on failure; a length below 1 is refused before oneMKL.
    pub fn bf_dftr_new(
        q: *mut c_void,
        prec: i32,
        ndim: i64,
        lengths: *const i64,
        fwd_scale: f64,
        bwd_scale: f64,
    ) -> *mut c_void;
    pub fn bf_dftr_free(h: *mut c_void);
    /// Blocking real transform. `dir` 0: real `input` to half-spectrum complex `out`
    /// (last dimension `n/2 + 1`); `dir` 1: the reverse.
    pub fn bf_dftr_compute(h: *mut c_void, dir: i32, input: *mut c_void, out: *mut c_void) -> i32;
    /// Non-blocking `bf_dftr_compute`: returns the event, or null on failure.
    pub fn bf_dftr_compute_async(
        h: *mut c_void,
        dir: i32,
        input: *mut c_void,
        out: *mut c_void,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}
