//! Complex-to-complex DFT through the shim on the CPU (OpenCL) device. Compiled out without
//! oneAPI; the oneAPI CI job sets `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};

fn err() -> String {
    unsafe { CStr::from_ptr(bf_last_error()) }
        .to_string_lossy()
        .into_owned()
}

struct Env {
    q: *mut c_void,
}
impl Env {
    fn new() -> Self {
        let q = unsafe { bf_queue_new(1) };
        assert!(!q.is_null(), "no CPU SYCL device: {}", err());
        Env { q }
    }
    /// Interleaved (re, im) f64 USM buffer holding `v`.
    fn usm(&self, v: &[f64]) -> *mut f64 {
        let p = unsafe { bf_malloc(self.q, 0, (v.len() * 8) as i64) }.cast::<f64>();
        assert!(!p.is_null(), "{}", err());
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len()) };
        p
    }
    fn plan(&self, dims: &[i64], inplace: bool, fwd: f64, bwd: f64) -> *mut c_void {
        let h = unsafe {
            bf_dft_c2c_new(
                self.q,
                1,
                dims.len() as i64,
                dims.as_ptr(),
                inplace as i32,
                fwd,
                bwd,
            )
        };
        assert!(!h.is_null(), "{}", err());
        h
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        unsafe { bf_queue_free(self.q) }
    }
}

fn read(p: *mut f64, n: usize) -> Vec<f64> {
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "element {i}: got {g}, want {w}\n got  {got:?}\n want {want:?}"
        );
    }
}

/// Textbook O(n^2) forward DFT on interleaved data, as an independent reference.
fn naive_dft(x: &[f64]) -> Vec<f64> {
    let n = x.len() / 2;
    let mut out = vec![0.0; 2 * n];
    for k in 0..n {
        for j in 0..n {
            let ang = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
            let (s, c) = ang.sin_cos();
            out[2 * k] += x[2 * j] * c - x[2 * j + 1] * s;
            out[2 * k + 1] += x[2 * j] * s + x[2 * j + 1] * c;
        }
    }
    out
}

#[test]
fn forward_matches_naive_dft_for_non_power_of_two() {
    let env = Env::new();
    let x: Vec<f64> = (0..12).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect(); // n = 6
    let buf = env.usm(&x);
    let h = env.plan(&[6], true, 1.0, 1.0);
    assert_eq!(
        unsafe { bf_dft_compute(h, 0, buf.cast(), std::ptr::null_mut()) },
        0,
        "{}",
        err()
    );
    assert_close(&read(buf, 12), &naive_dft(&x), 1e-12);
    unsafe { bf_dft_free(h) };
}

#[test]
fn delta_transforms_to_all_ones_and_constant_to_delta() {
    let env = Env::new();
    let delta = env.usm(&[1., 0., 0., 0., 0., 0., 0., 0.]);
    let h = env.plan(&[4], true, 1.0, 1.0);
    assert_eq!(
        unsafe { bf_dft_compute(h, 0, delta.cast(), std::ptr::null_mut()) },
        0
    );
    assert_close(&read(delta, 8), &[1., 0., 1., 0., 1., 0., 1., 0.], 1e-14);
    // Backward of all-ones is n * delta (unnormalised).
    assert_eq!(
        unsafe { bf_dft_compute(h, 1, delta.cast(), std::ptr::null_mut()) },
        0
    );
    assert_close(&read(delta, 8), &[4., 0., 0., 0., 0., 0., 0., 0.], 1e-14);
    unsafe { bf_dft_free(h) };
}

#[test]
fn two_dimensional_delta_is_all_ones() {
    let env = Env::new();
    let mut x = vec![0.0; 2 * 6];
    x[0] = 1.0;
    let buf = env.usm(&x);
    let h = env.plan(&[2, 3], true, 1.0, 1.0);
    assert_eq!(
        unsafe { bf_dft_compute(h, 0, buf.cast(), std::ptr::null_mut()) },
        0,
        "{}",
        err()
    );
    let want: Vec<f64> = (0..6).flat_map(|_| [1.0, 0.0]).collect();
    assert_close(&read(buf, 12), &want, 1e-14);
    unsafe { bf_dft_free(h) };
}

#[test]
fn scale_factors_are_honoured() {
    // Negative control: with the default scales a round trip returns n*x; with backward scale 1/n
    // it returns x. A shim that dropped the scale arguments would fail the second assertion.
    let env = Env::new();
    let x = [1.0, 2.0, -3.0, 0.5, 4.0, -1.0, 0.0, 2.5]; // n = 4
    for (bwd, factor) in [(1.0, 4.0), (0.25, 1.0)] {
        let buf = env.usm(&x);
        let h = env.plan(&[4], true, 1.0, bwd);
        assert_eq!(
            unsafe { bf_dft_compute(h, 0, buf.cast(), std::ptr::null_mut()) },
            0
        );
        assert_eq!(
            unsafe { bf_dft_compute(h, 1, buf.cast(), std::ptr::null_mut()) },
            0
        );
        let want: Vec<f64> = x.iter().map(|v| v * factor).collect();
        assert_close(&read(buf, 8), &want, 1e-12);
        unsafe { bf_dft_free(h) };
    }
}

#[test]
fn out_of_place_leaves_input_untouched() {
    let env = Env::new();
    let x = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let (input, output) = (env.usm(&x), env.usm(&[0.0; 8]));
    let h = env.plan(&[4], false, 1.0, 1.0);
    assert_eq!(
        unsafe { bf_dft_compute(h, 0, input.cast(), output.cast()) },
        0,
        "{}",
        err()
    );
    assert_close(&read(input, 8), &x, 0.0);
    assert_close(&read(output, 8), &naive_dft(&x), 1e-12);
    unsafe { bf_dft_free(h) };
}

#[test]
fn invalid_plan_is_an_error_not_a_crash() {
    let env = Env::new();
    // A zero length must be refused by the shim itself, before oneMKL: on Intel Xeon CPUs with
    // default dispatch oneMKL crashes with SIGFPE on it (ROADMAP R9). The message proves whose
    // refusal it is: oneMKL's own would be `FFT_INVALID_DESCRIPTOR`.
    for dims in [vec![0_i64], vec![4, 0], vec![0, 4], vec![-1], vec![4, -3]] {
        let h = unsafe { bf_dft_c2c_new(env.q, 1, dims.len() as i64, dims.as_ptr(), 1, 1.0, 1.0) };
        assert!(h.is_null(), "{dims:?} must be refused");
        let msg = err();
        assert!(
            msg.contains("every length must be at least 1"),
            "{dims:?}: refused by oneMKL or not at all, not by the shim: {msg:?}"
        );
    }
    // Control: a valid length is accepted by the same call.
    let ok_dims = [4_i64];
    let ok = unsafe { bf_dft_c2c_new(env.q, 1, 1, ok_dims.as_ptr(), 1, 1.0, 1.0) };
    assert!(!ok.is_null());
    unsafe { bf_dft_free(ok) };
    let none = unsafe { bf_dft_c2c_new(env.q, 1, 0, std::ptr::null(), 1, 1.0, 1.0) };
    assert!(none.is_null());
}

#[test]
fn single_precision_forward_matches_naive_dft() {
    let env = Env::new();
    let x: Vec<f32> = (0..12).map(|i| ((i * 5 + 1) % 9) as f32 - 4.0).collect(); // n = 6
    let p = unsafe { bf_malloc(env.q, 0, (x.len() * 4) as i64) }.cast::<f32>();
    assert!(!p.is_null(), "{}", err());
    unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), p, x.len()) };
    let dims = [6_i64];
    let h = unsafe { bf_dft_c2c_new(env.q, 0, 1, dims.as_ptr(), 1, 1.0, 1.0) };
    assert!(!h.is_null(), "{}", err());
    assert_eq!(
        unsafe { bf_dft_compute(h, 0, p.cast(), std::ptr::null_mut()) },
        0,
        "{}",
        err()
    );
    let got: Vec<f64> = unsafe { std::slice::from_raw_parts(p, 12) }
        .iter()
        .map(|v| *v as f64)
        .collect();
    let want = naive_dft(&x.iter().map(|v| *v as f64).collect::<Vec<_>>());
    assert_close(&got, &want, 1e-4);
    unsafe { bf_dft_free(h) };
    unsafe { bf_free(env.q, p.cast()) };
}
