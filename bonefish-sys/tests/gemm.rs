//! Runs oneMKL SYCL GEMM through the shim on the CPU (OpenCL) device. Compiled out without
//! oneAPI; the oneAPI CI job sets `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};

struct Usm {
    q: *mut c_void,
    p: *mut f32,
}
impl Usm {
    fn new(q: *mut c_void, v: &[f32]) -> Self {
        let p = unsafe { bf_malloc(q, 0, (v.len() * 4) as i64) }.cast::<f32>();
        assert!(!p.is_null(), "{}", err());
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len()) };
        Usm { q, p }
    }
    fn read(&self, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.p, n) }.to_vec()
    }
}
impl Drop for Usm {
    fn drop(&mut self) {
        unsafe { bf_free(self.q, self.p.cast()) }
    }
}

fn err() -> String {
    unsafe { CStr::from_ptr(bf_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn cpu_queue() -> *mut c_void {
    let q = unsafe { bf_queue_new(1) };
    assert!(!q.is_null(), "no CPU SYCL device: {}", err());
    q
}

#[test]
fn sgemm_matches_hand_computation() {
    let q = cpu_queue();
    // Column-major A = [1 3; 2 4], B = [0 0; 1 1]; A*B = [3 3; 4 4].
    let a = Usm::new(q, &[1., 2., 3., 4.]);
    let b = Usm::new(q, &[0., 1., 0., 1.]);
    let c = Usm::new(q, &[0.; 4]);
    let rc = unsafe { bf_sgemm(q, 0, 0, 2, 2, 2, 1.0, a.p, 2, b.p, 2, 0.0, c.p, 2) };
    assert_eq!(rc, 0, "{}", err());
    assert_eq!(c.read(4), vec![3., 4., 3., 4.]);
    drop((a, b, c));
    unsafe { bf_queue_free(q) };
}

#[test]
fn beta_accumulates_and_transpose_is_honoured() {
    // Negative control for the test above: a shim that ignored beta or the transpose flag would
    // not produce these numbers.
    let q = cpu_queue();
    let a = Usm::new(q, &[1., 2., 3., 4.]);
    let b = Usm::new(q, &[0., 1., 0., 1.]);
    let c = Usm::new(q, &[10.; 4]);
    // C = A^T * B + 1*C; A^T = [1 2; 3 4], so A^T*B = [2 2; 4 4], column-major 2 4 2 4.
    let rc = unsafe { bf_sgemm(q, 1, 0, 2, 2, 2, 1.0, a.p, 2, b.p, 2, 1.0, c.p, 2) };
    assert_eq!(rc, 0, "{}", err());
    assert_eq!(c.read(4), vec![12., 14., 12., 14.]);
    drop((a, b, c));
    unsafe { bf_queue_free(q) };
}

#[test]
fn bad_leading_dimension_is_an_error_not_a_crash() {
    let q = cpu_queue();
    let a = Usm::new(q, &[1., 2., 3., 4.]);
    let c = Usm::new(q, &[0.; 4]);
    let rc = unsafe { bf_sgemm(q, 0, 0, 2, 2, 2, 1.0, a.p, 1, a.p, 2, 0.0, c.p, 2) };
    assert_ne!(rc, 0);
    assert!(!err().is_empty());
    drop((a, c));
    unsafe { bf_queue_free(q) };
}

#[test]
fn reports_device_name() {
    let q = cpu_queue();
    let mut buf = [0 as std::ffi::c_char; 256];
    let n = unsafe { bf_queue_device_name(q, buf.as_mut_ptr(), 256) };
    assert!(n > 0);
    assert!(
        !unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_bytes()
            .is_empty()
    );
    unsafe { bf_queue_free(q) };
}
