//! R2 remainder: `iamin`, `rot`, `ger`, `rotm`, `symv`, `syr`, `syr2`, `trmv`, `trsv`, `gbmv` through the shim on the CPU (OpenCL) device, blocking
//! and non-blocking, single and double precision from one macro. Compiled out without oneAPI; the
//! oneAPI CI job sets `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
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

/// Shared USM array of `T` that frees itself.
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

macro_rules! ext_tests {
    ($m:ident, $t:ty, $iamin:ident, $rot:ident, $ger:ident, $iamin_a:ident, $rot_a:ident,
     $ger_a:ident) => {
        mod $m {
            use super::*;

            #[test]
            fn iamin_is_zero_based_and_first_on_ties() {
                let q = cpu_queue();
                let i = Usm::new(q, &[-1_i64]);
                let x = Usm::new(q, &[3.0 as $t, -1.0, 2.0, 1.0]);
                assert_eq!(unsafe { $iamin(q, 4, x.p, 1, i.p) }, 0, "{}", err());
                assert_eq!(i.read(), vec![1], "|-1| ties |1|; the first wins");
                let j = Usm::new(q, &[-1_i64]);
                wait_free(
                    unsafe { $iamin_a(q, 4, x.p, 1, j.p, std::ptr::null(), 0) },
                    "iamin",
                );
                assert_eq!(j.read(), vec![1]);
                drop((i, j, x));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn rot_is_a_givens_rotation() {
                let q = cpu_queue();
                // c = 0, s = 1: x' = y, y' = -x.
                let (x, y) = (
                    Usm::new(q, &[1.0 as $t, 2.0, 3.0]),
                    Usm::new(q, &[4.0 as $t, 5.0, 6.0]),
                );
                assert_eq!(
                    unsafe { $rot(q, 3, x.p, 1, y.p, 1, 0.0, 1.0) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(x.read(), vec![4.0, 5.0, 6.0]);
                assert_eq!(y.read(), vec![-1.0, -2.0, -3.0]);
                // The same through the non-blocking entry, strided: only elements 0 and 2.
                let (x2, y2) = (
                    Usm::new(q, &[1.0 as $t, 2.0, 3.0]),
                    Usm::new(q, &[4.0 as $t, 5.0, 6.0]),
                );
                let ev = unsafe { $rot_a(q, 2, x2.p, 2, y2.p, 2, 0.0, 1.0, std::ptr::null(), 0) };
                wait_free(ev, "rot");
                assert_eq!(x2.read(), vec![4.0, 2.0, 6.0]);
                assert_eq!(y2.read(), vec![-1.0, 5.0, -3.0]);
                drop((x, y, x2, y2));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn ger_is_a_rank_one_update() {
                let q = cpu_queue();
                // A (2 x 3, column-major) = alpha * x * y^T + A, with x = [1, 2], y = [1, 10, 100].
                let a = Usm::new(q, &[1.0 as $t; 6]);
                let x = Usm::new(q, &[1.0 as $t, 2.0]);
                let y = Usm::new(q, &[1.0 as $t, 10.0, 100.0]);
                assert_eq!(
                    unsafe { $ger(q, 2, 3, 2.0, x.p, 1, y.p, 1, a.p, 2) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(a.read(), vec![3.0, 5.0, 21.0, 41.0, 201.0, 401.0]);
                let b = Usm::new(q, &[0.0 as $t; 6]);
                let ev =
                    unsafe { $ger_a(q, 2, 3, 1.0, x.p, 1, y.p, 1, b.p, 2, std::ptr::null(), 0) };
                wait_free(ev, "ger");
                assert_eq!(b.read(), vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0]);
                // Error path: lda < m is rejected with a status, not undefined behaviour.
                let rc = unsafe { $ger(q, 2, 3, 1.0, x.p, 1, y.p, 1, b.p, 1) };
                assert_ne!(rc, 0, "lda = 1 < m = 2 must be rejected");
                assert!(!err().is_empty());
                drop((a, b, x, y));
                unsafe { bf_queue_free(q) };
            }
        }
    };
}

ext_tests!(
    single,
    f32,
    bf_siamin,
    bf_srot,
    bf_sger,
    bf_siamin_async,
    bf_srot_async,
    bf_sger_async
);
ext_tests!(
    double,
    f64,
    bf_diamin,
    bf_drot,
    bf_dger,
    bf_diamin_async,
    bf_drot_async,
    bf_dger_async
);

// R2 remainder: rotm, symv, syr, syr2, trmv, trsv, gbmv. Hand-computed values, column-major.
// `uplo`: 0 upper, 1 lower. The triangle that must not be touched or read holds a sentinel.
macro_rules! l2_tests {
    ($m:ident, $t:ty, $rotm:ident, $symv:ident, $syr:ident, $syr2:ident, $trmv:ident, $trsv:ident,
     $gbmv:ident, $rotm_a:ident, $symv_a:ident, $syr_a:ident, $syr2_a:ident, $trmv_a:ident,
     $trsv_a:ident, $gbmv_a:ident) => {
        mod $m {
            use super::*;
            const NO: *const *mut c_void = std::ptr::null();

            #[test]
            fn rotm_applies_the_full_h_matrix() {
                let q = cpu_queue();
                // flag -1: H = [[h11, h12], [h21, h22]] = [[1, 2], [3, 4]], stored h11 h21 h12 h22.
                let p = [-1.0 as $t, 1.0, 3.0, 2.0, 4.0];
                let (x, y, par) = (
                    Usm::new(q, &[1.0 as $t, 2.0]),
                    Usm::new(q, &[10.0 as $t, 20.0]),
                    Usm::new(q, &p),
                );
                assert_eq!(
                    unsafe { $rotm(q, 2, x.p, 1, y.p, 1, par.p) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(x.read(), vec![21.0, 42.0]);
                assert_eq!(y.read(), vec![43.0, 86.0]);
                let (x2, y2) = (
                    Usm::new(q, &[1.0 as $t, 2.0]),
                    Usm::new(q, &[10.0 as $t, 20.0]),
                );
                wait_free(
                    unsafe { $rotm_a(q, 2, x2.p, 1, y2.p, 1, par.p, NO, 0) },
                    "rotm",
                );
                assert_eq!((x2.read(), y2.read()), (x.read(), y.read()));
                drop((x, y, par, x2, y2));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn symv_reads_one_triangle_only() {
                let q = cpu_queue();
                // A = [[2, 1], [1, 3]]; the other triangle is 99 and must not be read.
                let x = Usm::new(q, &[1.0 as $t, 2.0]);
                let up = Usm::new(q, &[2.0 as $t, 99.0, 1.0, 3.0]);
                let y = Usm::new(q, &[10.0 as $t, 10.0]);
                assert_eq!(
                    unsafe { $symv(q, 0, 2, 1.0, up.p, 2, x.p, 1, 1.0, y.p, 1) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(y.read(), vec![14.0, 17.0]);
                let lo = Usm::new(q, &[2.0 as $t, 1.0, 99.0, 3.0]);
                let y2 = Usm::new(q, &[10.0 as $t, 10.0]);
                wait_free(
                    unsafe { $symv_a(q, 1, 2, 1.0, lo.p, 2, x.p, 1, 1.0, y2.p, 1, NO, 0) },
                    "symv",
                );
                assert_eq!(y2.read(), vec![14.0, 17.0]);
                drop((x, up, y, lo, y2));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn syr_updates_one_triangle() {
                let q = cpu_queue();
                let x = Usm::new(q, &[1.0 as $t, 2.0]);
                let a = Usm::new(q, &[7.0 as $t; 4]);
                assert_eq!(
                    unsafe { $syr(q, 0, 2, 2.0, x.p, 1, a.p, 2) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(a.read(), vec![9.0, 7.0, 11.0, 15.0]);
                let b = Usm::new(q, &[7.0 as $t; 4]);
                wait_free(
                    unsafe { $syr_a(q, 1, 2, 2.0, x.p, 1, b.p, 2, NO, 0) },
                    "syr",
                );
                assert_eq!(b.read(), vec![9.0, 11.0, 7.0, 15.0]);
                drop((x, a, b));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn syr2_updates_one_triangle() {
                let q = cpu_queue();
                // x y^T + y x^T = [[6, 10], [10, 16]] for x = [1, 2], y = [3, 4].
                let x = Usm::new(q, &[1.0 as $t, 2.0]);
                let y = Usm::new(q, &[3.0 as $t, 4.0]);
                let a = Usm::new(q, &[0.0 as $t; 4]);
                assert_eq!(
                    unsafe { $syr2(q, 0, 2, 1.0, x.p, 1, y.p, 1, a.p, 2) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(a.read(), vec![6.0, 0.0, 10.0, 16.0]);
                let b = Usm::new(q, &[0.0 as $t; 4]);
                wait_free(
                    unsafe { $syr2_a(q, 1, 2, 1.0, x.p, 1, y.p, 1, b.p, 2, NO, 0) },
                    "syr2",
                );
                assert_eq!(b.read(), vec![6.0, 10.0, 0.0, 16.0]);
                drop((x, y, a, b));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn trmv_and_trsv_honour_uplo_trans_and_diag() {
                let q = cpu_queue();
                // Upper triangular A = [[2, 3], [0, 4]]; the 99 below the diagonal is never read.
                let a = Usm::new(q, &[2.0 as $t, 99.0, 3.0, 4.0]);
                let ones = |q| Usm::new(q, &[1.0 as $t, 1.0]);
                // x = A * [1, 1] = [5, 4].
                let x = ones(q);
                assert_eq!(
                    unsafe { $trmv(q, 0, 0, 0, 2, a.p, 2, x.p, 1) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(x.read(), vec![5.0, 4.0]);
                // A^T * [1, 1] = [2, 7].
                let xt = ones(q);
                assert_eq!(
                    unsafe { $trmv(q, 0, 1, 0, 2, a.p, 2, xt.p, 1) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(xt.read(), vec![2.0, 7.0]);
                // Unit diagonal: [[1, 3], [0, 1]] * [1, 1] = [4, 1] (the stored 2 and 4 are ignored).
                let xu = ones(q);
                wait_free(
                    unsafe { $trmv_a(q, 0, 0, 1, 2, a.p, 2, xu.p, 1, NO, 0) },
                    "trmv",
                );
                assert_eq!(xu.read(), vec![4.0, 1.0]);
                // trsv inverts trmv: solve A x = [5, 4] gives [1, 1].
                let b = Usm::new(q, &[5.0 as $t, 4.0]);
                assert_eq!(
                    unsafe { $trsv(q, 0, 0, 0, 2, a.p, 2, b.p, 1) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(b.read(), vec![1.0, 1.0]);
                let bt = Usm::new(q, &[2.0 as $t, 7.0]);
                assert_eq!(
                    unsafe { $trsv(q, 0, 1, 0, 2, a.p, 2, bt.p, 1) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(bt.read(), vec![1.0, 1.0]);
                let bu = Usm::new(q, &[4.0 as $t, 1.0]);
                wait_free(
                    unsafe { $trsv_a(q, 0, 0, 1, 2, a.p, 2, bu.p, 1, NO, 0) },
                    "trsv",
                );
                assert_eq!(bu.read(), vec![1.0, 1.0]);
                // Lower: [[2, 0], [3, 4]] (stored column-major, the upper 99 never read).
                let l = Usm::new(q, &[2.0 as $t, 3.0, 99.0, 4.0]);
                let xl = ones(q);
                assert_eq!(
                    unsafe { $trmv(q, 1, 0, 0, 2, l.p, 2, xl.p, 1) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(xl.read(), vec![2.0, 7.0]);
                drop((a, x, xt, xu, b, bt, bu, l, xl));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn gbmv_reads_band_storage() {
                let q = cpu_queue();
                // M = [[1, 2, 0], [3, 4, 5], [0, 6, 7]], kl = ku = 1, lda = 3. Band column j holds
                // rows j-1..j+1 at index (ku + i - j): unused corners are 0.
                let a = Usm::new(q, &[0.0 as $t, 1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 7.0, 0.0]);
                let x = Usm::new(q, &[1.0 as $t, 1.0, 1.0]);
                let y = Usm::new(q, &[9.0 as $t; 3]);
                assert_eq!(
                    unsafe { $gbmv(q, 0, 3, 3, 1, 1, 1.0, a.p, 3, x.p, 1, 0.0, y.p, 1) },
                    0,
                    "{}",
                    err()
                );
                assert_eq!(y.read(), vec![3.0, 12.0, 13.0]);
                let yt = Usm::new(q, &[9.0 as $t; 3]);
                wait_free(
                    unsafe { $gbmv_a(q, 1, 3, 3, 1, 1, 1.0, a.p, 3, x.p, 1, 0.0, yt.p, 1, NO, 0) },
                    "gbmv",
                );
                assert_eq!(yt.read(), vec![4.0, 12.0, 12.0]);
                drop((a, x, y, yt));
                unsafe { bf_queue_free(q) };
            }
        }
    };
}

l2_tests!(
    l2_single,
    f32,
    bf_srotm,
    bf_ssymv,
    bf_ssyr,
    bf_ssyr2,
    bf_strmv,
    bf_strsv,
    bf_sgbmv,
    bf_srotm_async,
    bf_ssymv_async,
    bf_ssyr_async,
    bf_ssyr2_async,
    bf_strmv_async,
    bf_strsv_async,
    bf_sgbmv_async
);
l2_tests!(
    l2_double,
    f64,
    bf_drotm,
    bf_dsymv,
    bf_dsyr,
    bf_dsyr2,
    bf_dtrmv,
    bf_dtrsv,
    bf_dgbmv,
    bf_drotm_async,
    bf_dsymv_async,
    bf_dsyr_async,
    bf_dsyr2_async,
    bf_dtrmv_async,
    bf_dtrsv_async,
    bf_dgbmv_async
);
