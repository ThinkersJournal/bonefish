//! R2 remainder, complex level 3: `symm`, `hemm`, `syrk`, `herk`, `syr2k`, `her2k`, `trmm`, `trsm`
//! (non-blocking) through the shim on the CPU (OpenCL) device, single (`c`) and double (`z`)
//! precision from one macro, against hand-computed 2 x 2 results. The triangle a routine must not
//! touch is poisoned with 99. Compiled out without oneAPI; the oneAPI CI job sets
//! `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};
mod common;

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

macro_rules! cl3_tests {
    ($m:ident, $t:ty, $symm:ident, $hemm:ident, $syrk:ident, $herk:ident, $syr2k:ident,
     $her2k:ident, $trmm:ident, $trsm:ident) => {
        mod $m {
            use super::*;
            const NONE: *const *mut c_void = std::ptr::null();
            const ONE: [$t; 2] = [1.0, 0.0];
            const ZERO: [$t; 2] = [0.0, 0.0];

            /// Interleaved column-major complex data from `(re, im)` pairs.
            fn cx(x: &[(f64, f64)]) -> Vec<$t> {
                x.iter().flat_map(|&(r, i)| [r as $t, i as $t]).collect()
            }

            fn close(got: &[$t], want: &[(f64, f64)], what: &str) {
                let want = cx(want);
                assert_eq!(got.len(), want.len(), "{what}");
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-4,
                        "{what}: element {k}: {g} vs {w} in {got:?}"
                    );
                }
            }

            const P: (f64, f64) = (99.0, 99.0);

            #[test]
            fn trsm_and_trmm_left_upper() {
                // A = [[2, i], [0, 1]] (upper stored; the lower slot is poisoned).
                let a_v = cx(&[(2.0, 0.0), P, (0.0, 1.0), (1.0, 0.0)]);
                let q = cpu_queue();
                let a = Usm::new(q, &a_v);
                // Solve A X = B for B = [4, 1]: x1 = 1, x0 = (4 - i) / 2 = 2 - 0.5i.
                let b = Usm::new(q, &cx(&[(4.0, 0.0), (1.0, 0.0)]));
                let ev =
                    unsafe { $trsm(q, 0, 0, 0, 0, 2, 1, ONE.as_ptr(), a.p, 2, b.p, 2, NONE, 0) };
                wait_free(ev, "trsm");
                close(&b.read(), &[(2.0, -0.5), (1.0, 0.0)], "trsm");
                // B := A B for B = [1, 1]: [2 + i, 1].
                let b = Usm::new(q, &cx(&[(1.0, 0.0), (1.0, 0.0)]));
                let ev =
                    unsafe { $trmm(q, 0, 0, 0, 0, 2, 1, ONE.as_ptr(), a.p, 2, b.p, 2, NONE, 0) };
                wait_free(ev, "trmm");
                close(&b.read(), &[(2.0, 1.0), (1.0, 0.0)], "trmm");
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn symm_and_hemm_left_upper() {
                let q = cpu_queue();
                let b = Usm::new(q, &cx(&[(1.0, 0.0), (0.0, 0.0)]));
                // Symmetric A = [[2, 1+i], [1+i, 3]], upper stored: A B = first column.
                let a = Usm::new(q, &cx(&[(2.0, 0.0), P, (1.0, 1.0), (3.0, 0.0)]));
                let c = Usm::new(q, &cx(&[(0.0, 0.0), (0.0, 0.0)]));
                let ev = unsafe {
                    $symm(
                        q,
                        0,
                        0,
                        2,
                        1,
                        ONE.as_ptr(),
                        a.p,
                        2,
                        b.p,
                        2,
                        ZERO.as_ptr(),
                        c.p,
                        2,
                        NONE,
                        0,
                    )
                };
                wait_free(ev, "symm");
                close(&c.read(), &[(2.0, 0.0), (1.0, 1.0)], "symm");
                // Hermitian with the same upper triangle: the lower element is conj(1+i) = 1-i.
                let c = Usm::new(q, &cx(&[(0.0, 0.0), (0.0, 0.0)]));
                let ev = unsafe {
                    $hemm(
                        q,
                        0,
                        0,
                        2,
                        1,
                        ONE.as_ptr(),
                        a.p,
                        2,
                        b.p,
                        2,
                        ZERO.as_ptr(),
                        c.p,
                        2,
                        NONE,
                        0,
                    )
                };
                wait_free(ev, "hemm");
                close(&c.read(), &[(2.0, 0.0), (1.0, -1.0)], "hemm");
                unsafe { bf_queue_free(q) };
            }

            /// Rank-k updates of the upper triangle of a zero 2 x 2 C, lower slot poisoned.
            fn upd(
                want: [(f64, f64); 3],
                run: impl FnOnce(*mut c_void, &Usm<$t>, &Usm<$t>, &Usm<$t>) -> *mut c_void,
                what: &str,
            ) {
                let q = cpu_queue();
                // A = [1, i]^T, B = [1, 1]^T (2 x 1 each).
                let a = Usm::new(q, &cx(&[(1.0, 0.0), (0.0, 1.0)]));
                let b = Usm::new(q, &cx(&[(1.0, 0.0), (1.0, 0.0)]));
                let c = Usm::new(q, &cx(&[(0.0, 0.0), P, (0.0, 0.0), (0.0, 0.0)]));
                wait_free(run(q, &a, &b, &c), what);
                // c00, c01, c11 are computed; c10 (index 1) keeps its poison.
                let got = c.read();
                close(&[got[0], got[1]], &[want[0]], what);
                close(&got[4..6], &[want[1]], what);
                close(&got[6..8], &[want[2]], what);
                close(&got[2..4], &[P], &format!("{what}: untouched triangle"));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn rank_k_updates_upper() {
                // A A^T = [[1, i], [i, -1]].
                upd(
                    [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)],
                    |q, a, _, c| unsafe {
                        $syrk(
                            q,
                            0,
                            0,
                            2,
                            1,
                            ONE.as_ptr(),
                            a.p,
                            2,
                            ZERO.as_ptr(),
                            c.p,
                            2,
                            NONE,
                            0,
                        )
                    },
                    "syrk",
                );
                // A A^H = [[1, -i], [i, 1]].
                upd(
                    [(1.0, 0.0), (0.0, -1.0), (1.0, 0.0)],
                    |q, a, _, c| unsafe { $herk(q, 0, 0, 2, 1, 1.0, a.p, 2, 0.0, c.p, 2, NONE, 0) },
                    "herk",
                );
                // A B^T + B A^T = [[2, 1+i], [1+i, 2i]].
                upd(
                    [(2.0, 0.0), (1.0, 1.0), (0.0, 2.0)],
                    |q, a, b, c| unsafe {
                        $syr2k(
                            q,
                            0,
                            0,
                            2,
                            1,
                            ONE.as_ptr(),
                            a.p,
                            2,
                            b.p,
                            2,
                            ZERO.as_ptr(),
                            c.p,
                            2,
                            NONE,
                            0,
                        )
                    },
                    "syr2k",
                );
                // A B^H + B A^H = [[2, 1-i], [1+i, 0]].
                upd(
                    [(2.0, 0.0), (1.0, -1.0), (0.0, 0.0)],
                    |q, a, b, c| unsafe {
                        $her2k(
                            q,
                            0,
                            0,
                            2,
                            1,
                            ONE.as_ptr(),
                            a.p,
                            2,
                            b.p,
                            2,
                            0.0,
                            c.p,
                            2,
                            NONE,
                            0,
                        )
                    },
                    "her2k",
                );
            }
        }
    };
}

cl3_tests!(
    single,
    f32,
    bf_csymm_async,
    bf_chemm_async,
    bf_csyrk_async,
    bf_cherk_async,
    bf_csyr2k_async,
    bf_cher2k_async,
    bf_ctrmm_async,
    bf_ctrsm_async
);
cl3_tests!(
    double,
    f64,
    bf_zsymm_async,
    bf_zhemm_async,
    bf_zsyrk_async,
    bf_zherk_async,
    bf_zsyr2k_async,
    bf_zher2k_async,
    bf_ztrmm_async,
    bf_ztrsm_async
);
