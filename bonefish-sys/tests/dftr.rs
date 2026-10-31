//! Real-domain DFT through the shim (R3b): out-of-place real <-> half-spectrum complex. Needs
//! oneAPI at build time, so the file is empty without it. CPU/OpenCL device only; the data layout
//! below is oneMKL's default for an out-of-place real descriptor, checked here against a naive DFT
//! and **UNVERIFIED on a GPU**.
//!
//! Layout under test: forward reads `prod(n)` reals (row-major, last dimension contiguous) and
//! writes complex entries with the last dimension `n_last / 2 + 1` long; backward is the reverse.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::f64::consts::PI;
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
    fn usm<T: Copy>(&self, data: &[T]) -> *mut T {
        let bytes = (std::mem::size_of_val(data)).max(1) as i64;
        let p = unsafe { bf_malloc(self.q, 0, bytes) }.cast::<T>();
        assert!(!p.is_null());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
        p
    }
    fn plan(&self, prec: i32, dims: &[i64], fwd: f64, bwd: f64) -> *mut c_void {
        let h = unsafe { bf_dftr_new(self.q, prec, dims.len() as i64, dims.as_ptr(), fwd, bwd) };
        assert!(!h.is_null(), "bf_dftr_new {dims:?}: {}", err());
        h
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        unsafe { bf_queue_free(self.q) }
    }
}

fn read<T: Copy>(p: *const T, n: usize) -> Vec<T> {
    (0..n).map(|i| unsafe { *p.add(i) }).collect()
}

/// A real signal with no symmetry, so a transposed or reversed layout cannot pass by accident.
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0.37 * i as f64).sin() + 0.25 * ((i * i) % 7) as f64 - 0.1 * i as f64)
        .collect()
}

/// Naive half spectrum of a row-major real array with dims `dims`: complex (re, im) pairs in
/// row-major order with the last dimension `dims[d-1] / 2 + 1` long.
fn naive_half_spectrum(x: &[f64], dims: &[usize]) -> Vec<(f64, f64)> {
    let d = dims.len();
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .map(|(i, n)| if i == d - 1 { n / 2 + 1 } else { *n })
        .collect();
    let out_len: usize = out_dims.iter().product();
    let mut out = vec![(0.0, 0.0); out_len];
    for (oi, o) in out.iter_mut().enumerate() {
        // Decode the output multi-index.
        let mut rem = oi;
        let mut k = vec![0usize; d];
        for a in (0..d).rev() {
            k[a] = rem % out_dims[a];
            rem /= out_dims[a];
        }
        let (mut re, mut im) = (0.0, 0.0);
        for (xi, v) in x.iter().enumerate() {
            let mut rem = xi;
            let mut phase = 0.0;
            for a in (0..d).rev() {
                let n = rem % dims[a];
                rem /= dims[a];
                phase += (k[a] * n) as f64 / dims[a] as f64;
            }
            let ang = -2.0 * PI * phase;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        *o = (re, im);
    }
    out
}

fn max_err(got: &[f64], want: &[(f64, f64)]) -> f64 {
    assert_eq!(got.len(), 2 * want.len(), "length mismatch");
    want.iter()
        .enumerate()
        .map(|(i, (re, im))| (got[2 * i] - re).abs().max((got[2 * i + 1] - im).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn forward_matches_the_naive_half_spectrum_in_1d() {
    let env = Env::new();
    for n in [1_usize, 2, 3, 4, 5, 7, 8, 12, 16] {
        let x = signal(n);
        let h = env.plan(1, &[n as i64], 1.0, 1.0);
        let half = n / 2 + 1;
        let input = env.usm(&x);
        let out = env.usm(&vec![0.0_f64; 2 * half]);
        let rc = unsafe { bf_dftr_compute(h, 0, input.cast(), out.cast()) };
        assert_eq!(rc, 0, "n = {n}: {}", err());
        let got = read(out, 2 * half);
        let e = max_err(&got, &naive_half_spectrum(&x, &[n]));
        assert!(e < 1e-9, "n = {n}: max error {e}");
        unsafe { bf_dftr_free(h) };
    }
}

#[test]
fn single_precision_forward_matches_within_float_tolerance() {
    let env = Env::new();
    let n = 6;
    let x = signal(n);
    let xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    let h = env.plan(0, &[n as i64], 1.0, 1.0);
    let input = env.usm(&xf);
    let out = env.usm(&vec![0.0_f32; 2 * (n / 2 + 1)]);
    assert_eq!(
        unsafe { bf_dftr_compute(h, 0, input.cast(), out.cast()) },
        0,
        "{}",
        err()
    );
    let got: Vec<f64> = read(out, 2 * (n / 2 + 1))
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    let e = max_err(&got, &naive_half_spectrum(&x, &[n]));
    assert!(e < 1e-4, "max error {e}");
    unsafe { bf_dftr_free(h) };
}

#[test]
fn two_dimensional_layout_is_row_major_with_the_last_dimension_halved() {
    let env = Env::new();
    for dims in [[4_usize, 6], [5, 3], [3, 8]] {
        let n: usize = dims.iter().product();
        let x = signal(n);
        let h = env.plan(1, &[dims[0] as i64, dims[1] as i64], 1.0, 1.0);
        let out_len = dims[0] * (dims[1] / 2 + 1);
        let input = env.usm(&x);
        let out = env.usm(&vec![0.0_f64; 2 * out_len]);
        assert_eq!(
            unsafe { bf_dftr_compute(h, 0, input.cast(), out.cast()) },
            0,
            "{}",
            err()
        );
        let got = read(out, 2 * out_len);
        let want = naive_half_spectrum(&x, &dims);
        let e = max_err(&got, &want);
        assert!(e < 1e-9, "{dims:?}: max error {e}");
        unsafe { bf_dftr_free(h) };
    }
}

/// The negative control for the layout check: the same 4 x 6 result compared under the WRONG
/// layout (halving the first dimension instead of the last) must show a large error, so the check
/// above cannot pass by comparing the wrong thing.
#[test]
fn comparing_under_a_wrong_layout_is_detected() {
    let env = Env::new();
    let dims = [4_usize, 6];
    let x = signal(24);
    let h = env.plan(1, &[4, 6], 1.0, 1.0);
    let out_len = 4 * (6 / 2 + 1);
    let input = env.usm(&x);
    let out = env.usm(&vec![0.0_f64; 2 * out_len]);
    assert_eq!(
        unsafe { bf_dftr_compute(h, 0, input.cast(), out.cast()) },
        0,
        "{}",
        err()
    );
    let got = read(out, 2 * out_len);
    // Right layout: small error (control). Wrong layout (transpose the reference): large.
    let right = max_err(&got, &naive_half_spectrum(&x, &dims));
    let want = naive_half_spectrum(&x, &dims);
    let (rows, cols) = (4, 6 / 2 + 1);
    let mut transposed = vec![(0.0, 0.0); want.len()];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = want[r * cols + c];
        }
    }
    let wrong = max_err(&got, &transposed);
    assert!(right < 1e-9, "control failed: {right}");
    assert!(
        wrong > 1e-3,
        "a transposed layout was not detected: {wrong}"
    );
    unsafe { bf_dftr_free(h) };
}

#[test]
fn backward_undoes_forward_up_to_n_and_exactly_with_a_scale() {
    let env = Env::new();
    for n in [4_usize, 7, 12] {
        let x = signal(n);
        let half = n / 2 + 1;
        // Default scales: backward(forward(x)) = n * x.
        let h = env.plan(1, &[n as i64], 1.0, 1.0);
        let input = env.usm(&x);
        let spec = env.usm(&vec![0.0_f64; 2 * half]);
        let back = env.usm(&vec![0.0_f64; n]);
        assert_eq!(
            unsafe { bf_dftr_compute(h, 0, input.cast(), spec.cast()) },
            0,
            "{}",
            err()
        );
        assert_eq!(
            unsafe { bf_dftr_compute(h, 1, spec.cast(), back.cast()) },
            0,
            "{}",
            err()
        );
        for (i, (g, w)) in read(back, n).iter().zip(&x).enumerate() {
            assert!(
                (g - w * n as f64).abs() < 1e-9,
                "n = {n}, element {i}: {g} vs {}",
                w * n as f64
            );
        }
        unsafe { bf_dftr_free(h) };
        // With backward scale 1/n it is exact.
        let h = env.plan(1, &[n as i64], 1.0, 1.0 / n as f64);
        assert_eq!(
            unsafe { bf_dftr_compute(h, 0, input.cast(), spec.cast()) },
            0,
            "{}",
            err()
        );
        assert_eq!(
            unsafe { bf_dftr_compute(h, 1, spec.cast(), back.cast()) },
            0,
            "{}",
            err()
        );
        for (i, (g, w)) in read(back, n).iter().zip(&x).enumerate() {
            assert!(
                (g - w).abs() < 1e-9,
                "scaled, n = {n}, element {i}: {g} vs {w}"
            );
        }
        unsafe { bf_dftr_free(h) };
    }
}

#[test]
fn the_forward_scale_is_honoured() {
    let env = Env::new();
    let n = 8;
    let x = signal(n);
    let half = n / 2 + 1;
    let (plain, scaled) = (env.plan(1, &[8], 1.0, 1.0), env.plan(1, &[8], 3.0, 1.0));
    let input = env.usm(&x);
    let (a, b) = (
        env.usm(&vec![0.0_f64; 2 * half]),
        env.usm(&vec![0.0_f64; 2 * half]),
    );
    assert_eq!(
        unsafe { bf_dftr_compute(plain, 0, input.cast(), a.cast()) },
        0,
        "{}",
        err()
    );
    assert_eq!(
        unsafe { bf_dftr_compute(scaled, 0, input.cast(), b.cast()) },
        0,
        "{}",
        err()
    );
    for (i, (p, s)) in read(a, 2 * half).iter().zip(read(b, 2 * half)).enumerate() {
        assert!((s - 3.0 * p).abs() < 1e-9, "entry {i}: {s} vs {}", 3.0 * p);
    }
    unsafe {
        bf_dftr_free(plain);
        bf_dftr_free(scaled);
    }
}

/// A length below 1 must be refused by the shim itself (on Intel Xeon CPUs oneMKL kills the
/// process on a zero length, ROADMAP R9). The message proves whose refusal it is.
#[test]
fn bad_lengths_are_refused_by_the_shim_not_by_onemkl() {
    let env = Env::new();
    for dims in [vec![0_i64], vec![4, 0], vec![0, 4], vec![-1], vec![4, -3]] {
        let h = unsafe { bf_dftr_new(env.q, 1, dims.len() as i64, dims.as_ptr(), 1.0, 1.0) };
        assert!(h.is_null(), "{dims:?} must be refused");
        let msg = err();
        assert!(
            msg.contains("every length must be at least 1"),
            "{dims:?}: refused by oneMKL or not at all, not by the shim: {msg:?}"
        );
    }
    // Control: a valid length is accepted by the same call.
    let ok = env.plan(1, &[4], 1.0, 1.0);
    unsafe { bf_dftr_free(ok) };
    assert!(unsafe { bf_dftr_new(env.q, 1, 0, std::ptr::null(), 1.0, 1.0) }.is_null());
}

/// The async form gives the same answer after a wait.
#[test]
fn the_async_form_matches_the_blocking_one() {
    let env = Env::new();
    let n = 12;
    let x = signal(n);
    let half = n / 2 + 1;
    let h = env.plan(1, &[n as i64], 1.0, 1.0);
    let input = env.usm(&x);
    let out = env.usm(&vec![0.0_f64; 2 * half]);
    let e = unsafe { bf_dftr_compute_async(h, 0, input.cast(), out.cast(), std::ptr::null(), 0) };
    assert!(!e.is_null(), "{}", err());
    assert_eq!(unsafe { bf_event_wait(e) }, 0, "{}", err());
    unsafe { bf_event_free(e) };
    let got = read(out, 2 * half);
    assert!(max_err(&got, &naive_half_spectrum(&x, &[n])) < 1e-9);
    unsafe { bf_dftr_free(h) };
}

/// Backward in two dimensions: forward then backward gives `prod(n) * x`, which exercises the
/// explicit complex (backward-domain) strides as well as the real ones.
#[test]
fn two_dimensional_round_trip_returns_n_times_the_input() {
    let env = Env::new();
    for dims in [[4_usize, 6], [5, 3]] {
        let n: usize = dims.iter().product();
        let x = signal(n);
        let h = env.plan(1, &[dims[0] as i64, dims[1] as i64], 1.0, 1.0);
        let out_len = dims[0] * (dims[1] / 2 + 1);
        let input = env.usm(&x);
        let spec = env.usm(&vec![0.0_f64; 2 * out_len]);
        let back = env.usm(&vec![0.0_f64; n]);
        assert_eq!(
            unsafe { bf_dftr_compute(h, 0, input.cast(), spec.cast()) },
            0,
            "{}",
            err()
        );
        assert_eq!(
            unsafe { bf_dftr_compute(h, 1, spec.cast(), back.cast()) },
            0,
            "{}",
            err()
        );
        for (i, (g, w)) in read(back, n).iter().zip(&x).enumerate() {
            assert!(
                (g - w * n as f64).abs() < 1e-8,
                "{dims:?} element {i}: {g} vs {}",
                w * n as f64
            );
        }
        unsafe { bf_dftr_free(h) };
    }
}

/// Three dimensions, where the stride loop has more than one inner step to get right.
#[test]
fn three_dimensional_forward_matches_the_naive_half_spectrum() {
    let env = Env::new();
    for dims in [[2_usize, 3, 4], [3, 2, 5]] {
        let n: usize = dims.iter().product();
        let x = signal(n);
        let h = env.plan(
            1,
            &[dims[0] as i64, dims[1] as i64, dims[2] as i64],
            1.0,
            1.0,
        );
        let out_len = dims[0] * dims[1] * (dims[2] / 2 + 1);
        let input = env.usm(&x);
        let out = env.usm(&vec![0.0_f64; 2 * out_len]);
        assert_eq!(
            unsafe { bf_dftr_compute(h, 0, input.cast(), out.cast()) },
            0,
            "{}",
            err()
        );
        let e = max_err(&read(out, 2 * out_len), &naive_half_spectrum(&x, &dims));
        assert!(e < 1e-9, "{dims:?}: max error {e}");
        unsafe { bf_dftr_free(h) };
    }
}
