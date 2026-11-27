// SPDX-License-Identifier: MIT OR Apache-2.0
//! R2 remainder, level 3: `symm`, `syrk`, `syr2k`, `trmm`, `trsm` through the shim on the CPU
//! (OpenCL) device, blocking and non-blocking, single and double precision from one macro. The
//! triangle a routine must not touch is poisoned with 99. Compiled out without oneAPI; the oneAPI
//! CI job sets `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};
mod common;

const P: f64 = 99.0;

fn err() -> String {
    unsafe { CStr::from_ptr(bf_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn cpu_queue() -> *mut c_void {
    common::arm();
    let q = unsafe { bf_queue_new(1) };
    assert!(!q.is_null(), "no CPU SYCL device: {}", err());
    q
}

struct Usm<T: Copy> {
    q: *mut c_void,
    p: *mut T,
    n: usize,
}
impl<T: Copy> Usm<T> {
    fn new(q: *mut c_void, v: &[T]) -> Self {
        let bytes = (v.len().max(1) * std::mem::size_of::<T>()) as i64;
        let p = unsafe { bf_malloc(q, 0, bytes) }.cast::<T>();
        assert!(!p.is_null(), "{}", err());
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len()) };
        Usm { q, p, n: v.len() }
    }
    fn read(&self) -> Vec<T> {
        unsafe { std::slice::from_raw_parts(self.p, self.n) }.to_vec()
    }
}
impl<T: Copy> Drop for Usm<T> {
    fn drop(&mut self) {
        unsafe { bf_free(self.q, self.p.cast()) }
    }
}

fn wait_free(ev: *mut c_void, what: &str) {
    assert!(!ev.is_null(), "{what}: no event: {}", err());
    assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{what}: {}", err());
    unsafe { bf_event_free(ev) };
}

macro_rules! l3_tests {
    ($m:ident, $t:ty, $symm:ident, $syrk:ident, $syr2k:ident, $trmm:ident, $trsm:ident,
     $symm_a:ident, $syrk_a:ident, $syr2k_a:ident, $trmm_a:ident, $trsm_a:ident) => {
        mod $m {
            use super::*;
            const P: $t = super::P as $t;
            const NONE: *const *mut c_void = std::ptr::null();

            /// Column-major 2 x 2 helpers: `v(&[r0c0, r1c0, r0c1, r1c1])`.
            fn v(x: &[f64]) -> Vec<$t> {
                x.iter().map(|&e| e as $t).collect()
            }

            #[test]
            fn symm_left_upper_and_right_lower() {
                // A = [[2,1],[1,3]] stored in one triangle only; B = [[1,2],[3,4]].
                for async_ in [false, true] {
                    let q = cpu_queue();
                    let b = Usm::new(q, &v(&[1.0, 3.0, 2.0, 4.0]));
                    // Left, upper: C = A B = [[5,8],[10,14]].
                    let a = Usm::new(q, &[2.0 as $t, P, 1.0, 3.0]);
                    let c = Usm::new(q, &[0.0 as $t; 4]);
                    if async_ {
                        wait_free(
                            unsafe {
                                $symm_a(q, 0, 0, 2, 2, 1.0, a.p, 2, b.p, 2, 0.0, c.p, 2, NONE, 0)
                            },
                            "symm",
                        );
                    } else {
                        assert_eq!(
                            unsafe { $symm(q, 0, 0, 2, 2, 1.0, a.p, 2, b.p, 2, 0.0, c.p, 2) },
                            0,
                            "{}",
                            err()
                        );
                    }
                    assert_eq!(c.read(), v(&[5.0, 10.0, 8.0, 14.0]));
                    // Right, lower: C = B A = [[4,7],[10,15]].
                    let a2 = Usm::new(q, &[2.0 as $t, 1.0, P, 3.0]);
                    let c2 = Usm::new(q, &[0.0 as $t; 4]);
                    if async_ {
                        wait_free(
                            unsafe {
                                $symm_a(q, 1, 1, 2, 2, 1.0, a2.p, 2, b.p, 2, 0.0, c2.p, 2, NONE, 0)
                            },
                            "symm",
                        );
                    } else {
                        assert_eq!(
                            unsafe { $symm(q, 1, 1, 2, 2, 1.0, a2.p, 2, b.p, 2, 0.0, c2.p, 2) },
                            0,
                            "{}",
                            err()
                        );
                    }
                    assert_eq!(c2.read(), v(&[4.0, 10.0, 7.0, 15.0]));
                    drop((a, a2, b, c, c2));
                    unsafe { bf_queue_free(q) };
                }
            }

            #[test]
            fn syrk_writes_only_the_named_triangle() {
                // A = [1; 2] (2 x 1): A A^T = [[1,2],[2,4]]; upper written, lower stays poison.
                for async_ in [false, true] {
                    let q = cpu_queue();
                    let a = Usm::new(q, &v(&[1.0, 2.0]));
                    let c = Usm::new(q, &[P; 4]);
                    if async_ {
                        wait_free(
                            unsafe { $syrk_a(q, 0, 0, 2, 1, 1.0, a.p, 2, 0.0, c.p, 2, NONE, 0) },
                            "syrk",
                        );
                    } else {
                        assert_eq!(
                            unsafe { $syrk(q, 0, 0, 2, 1, 1.0, a.p, 2, 0.0, c.p, 2) },
                            0,
                            "{}",
                            err()
                        );
                    }
                    assert_eq!(c.read(), vec![1.0, P, 2.0, 4.0]);
                    drop((a, c));
                    unsafe { bf_queue_free(q) };
                }
            }

            #[test]
            fn syr2k_writes_only_the_named_triangle() {
                // A = [1; 2], B = [3; 4]: A B^T + B A^T = [[6,10],[10,16]]; upper only.
                for async_ in [false, true] {
                    let q = cpu_queue();
                    let a = Usm::new(q, &v(&[1.0, 2.0]));
                    let b = Usm::new(q, &v(&[3.0, 4.0]));
                    let c = Usm::new(q, &[P; 4]);
                    if async_ {
                        wait_free(
                            unsafe {
                                $syr2k_a(q, 0, 0, 2, 1, 1.0, a.p, 2, b.p, 2, 0.0, c.p, 2, NONE, 0)
                            },
                            "syr2k",
                        );
                    } else {
                        assert_eq!(
                            unsafe { $syr2k(q, 0, 0, 2, 1, 1.0, a.p, 2, b.p, 2, 0.0, c.p, 2) },
                            0,
                            "{}",
                            err()
                        );
                    }
                    assert_eq!(c.read(), vec![6.0, P, 10.0, 16.0]);
                    drop((a, b, c));
                    unsafe { bf_queue_free(q) };
                }
            }

            #[test]
            fn trmm_and_trsm_invert_each_other() {
                // A = [[2,1],[P,3]] upper, non-unit; B = [[1,2],[3,4]].
                // trmm: B := A B = [[5,8],[9,12]]; trsm solves it back.
                for async_ in [false, true] {
                    let q = cpu_queue();
                    let a = Usm::new(q, &[2.0 as $t, P, 1.0, 3.0]);
                    let b = Usm::new(q, &v(&[1.0, 3.0, 2.0, 4.0]));
                    if async_ {
                        wait_free(
                            unsafe { $trmm_a(q, 0, 0, 0, 0, 2, 2, 1.0, a.p, 2, b.p, 2, NONE, 0) },
                            "trmm",
                        );
                    } else {
                        assert_eq!(
                            unsafe { $trmm(q, 0, 0, 0, 0, 2, 2, 1.0, a.p, 2, b.p, 2) },
                            0,
                            "{}",
                            err()
                        );
                    }
                    assert_eq!(b.read(), v(&[5.0, 9.0, 8.0, 12.0]));
                    if async_ {
                        wait_free(
                            unsafe { $trsm_a(q, 0, 0, 0, 0, 2, 2, 1.0, a.p, 2, b.p, 2, NONE, 0) },
                            "trsm",
                        );
                    } else {
                        assert_eq!(
                            unsafe { $trsm(q, 0, 0, 0, 0, 2, 2, 1.0, a.p, 2, b.p, 2) },
                            0,
                            "{}",
                            err()
                        );
                    }
                    assert_eq!(b.read(), v(&[1.0, 3.0, 2.0, 4.0]));
                    // Unit diagonal ignores the stored 2 and 3: A = [[1,1],[0,1]], B := A B = [[4,6],[3,4]].
                    let b2 = Usm::new(q, &v(&[1.0, 3.0, 2.0, 4.0]));
                    assert_eq!(
                        unsafe { $trmm(q, 0, 0, 0, 1, 2, 2, 1.0, a.p, 2, b2.p, 2) },
                        0,
                        "{}",
                        err()
                    );
                    assert_eq!(b2.read(), v(&[4.0, 3.0, 6.0, 4.0]));
                    drop((a, b, b2));
                    unsafe { bf_queue_free(q) };
                }
            }
        }
    };
}

l3_tests!(
    f32_,
    f32,
    bf_ssymm,
    bf_ssyrk,
    bf_ssyr2k,
    bf_strmm,
    bf_strsm,
    bf_ssymm_async,
    bf_ssyrk_async,
    bf_ssyr2k_async,
    bf_strmm_async,
    bf_strsm_async
);
l3_tests!(
    f64_,
    f64,
    bf_dsymm,
    bf_dsyrk,
    bf_dsyr2k,
    bf_dtrmm,
    bf_dtrsm,
    bf_dsymm_async,
    bf_dsyrk_async,
    bf_dsyr2k_async,
    bf_dtrmm_async,
    bf_dtrsm_async
);
