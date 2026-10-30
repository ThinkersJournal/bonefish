//! BLAS level 1 and gemv through the shim on the CPU (OpenCL) device, single and double
//! precision from one macro so both run identical cases. Compiled out without oneAPI; the oneAPI
//! CI job sets `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};

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

macro_rules! blas1_tests {
    ($m:ident, $t:ty, $axpy:ident, $scal:ident, $copy:ident, $swap:ident, $dot:ident,
     $nrm2:ident, $asum:ident, $iamax:ident, $gemv:ident) => {
        mod $m {
            use super::*;

            fn xs() -> Vec<$t> {
                vec![1.0, -2.0, 3.0, -4.0]
            }
            fn ys() -> Vec<$t> {
                vec![10.0, 20.0, 30.0, 40.0]
            }

            #[test]
            fn axpy_adds_and_respects_stride() {
                let q = cpu_queue();
                let (x, y) = (Usm::new(q, &xs()), Usm::new(q, &ys()));
                assert_eq!(unsafe { $axpy(q, 4, 2.0, x.p, 1, y.p, 1) }, 0, "{}", err());
                assert_eq!(y.read(), vec![12.0, 16.0, 36.0, 32.0]);
                // Strided: n = 2, incx = 2, incy = 2 touches elements 0 and 2 only.
                let y2 = Usm::new(q, &ys());
                assert_eq!(unsafe { $axpy(q, 2, 1.0, x.p, 2, y2.p, 2) }, 0, "{}", err());
                assert_eq!(y2.read(), vec![11.0, 20.0, 33.0, 40.0]);
                drop((x, y, y2));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn scal_copy_swap() {
                let q = cpu_queue();
                let x = Usm::new(q, &xs());
                assert_eq!(unsafe { $scal(q, 4, -1.0, x.p, 1) }, 0);
                assert_eq!(x.read(), vec![-1.0, 2.0, -3.0, 4.0]);
                let (a, b) = (Usm::new(q, &xs()), Usm::new(q, &ys()));
                assert_eq!(unsafe { $copy(q, 4, a.p, 1, b.p, 1) }, 0);
                assert_eq!(b.read(), xs());
                let (c, d) = (Usm::new(q, &xs()), Usm::new(q, &ys()));
                assert_eq!(unsafe { $swap(q, 4, c.p, 1, d.p, 1) }, 0);
                assert_eq!((c.read(), d.read()), (ys(), xs()));
                drop((x, a, b, c, d));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn reductions_match_hand_values() {
                let q = cpu_queue();
                let (x, y) = (Usm::new(q, &xs()), Usm::new(q, &ys()));
                let r = Usm::new(q, &[0.0 as $t]);
                // 1*10 - 2*20 + 3*30 - 4*40 = -100; strided picks elements 0 and 2: 10 + 90.
                assert_eq!(unsafe { $dot(q, 4, x.p, 1, y.p, 1, r.p) }, 0, "{}", err());
                assert_eq!(r.read(), vec![-100.0]);
                assert_eq!(unsafe { $dot(q, 2, x.p, 2, y.p, 2, r.p) }, 0);
                assert_eq!(r.read(), vec![100.0]);
                let v = Usm::new(q, &[3.0 as $t, 4.0, 0.0, 0.0]);
                assert_eq!(unsafe { $nrm2(q, 4, v.p, 1, r.p) }, 0);
                assert_eq!(r.read(), vec![5.0]);
                assert_eq!(unsafe { $asum(q, 4, x.p, 1, r.p) }, 0);
                assert_eq!(r.read(), vec![10.0]);
                drop((x, y, r, v));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn iamax_is_zero_based_and_first_on_ties() {
                let q = cpu_queue();
                let i = Usm::new(q, &[-1_i64]);
                let x = Usm::new(q, &xs());
                assert_eq!(unsafe { $iamax(q, 4, x.p, 1, i.p) }, 0, "{}", err());
                assert_eq!(i.read(), vec![3], "|-4| is the largest, at index 3");
                let t = Usm::new(q, &[2.0 as $t, -2.0, 1.0]);
                assert_eq!(unsafe { $iamax(q, 3, t.p, 1, i.p) }, 0);
                assert_eq!(i.read(), vec![0], "ties resolve to the first");
                drop((i, x, t));
                unsafe { bf_queue_free(q) };
            }

            #[test]
            fn gemv_both_orientations() {
                let q = cpu_queue();
                // A = [1 3 5; 2 4 6], column-major 2 x 3.
                let a = Usm::new(q, &[1.0 as $t, 2.0, 3.0, 4.0, 5.0, 6.0]);
                let x = Usm::new(q, &[1.0 as $t, 1.0, 1.0]);
                let y = Usm::new(q, &[10.0 as $t, 10.0]);
                // y = 2*A*x + 1*y = 2*[9, 12] + [10, 10]
                assert_eq!(
                    unsafe { $gemv(q, 0, 2, 3, 2.0, a.p, 2, x.p, 1, 1.0, y.p, 1) },
                    0
                );
                assert_eq!(y.read(), vec![28.0, 34.0]);
                // y = A^T * [1, 1] = [3, 7, 11], beta 0 ignores the NaN already in y.
                let x2 = Usm::new(q, &[1.0 as $t, 1.0]);
                let y2 = Usm::new(q, &[<$t>::NAN; 3]);
                assert_eq!(
                    unsafe { $gemv(q, 1, 2, 3, 1.0, a.p, 2, x2.p, 1, 0.0, y2.p, 1) },
                    0
                );
                assert_eq!(y2.read(), vec![3.0, 7.0, 11.0]);
                drop((a, x, y, x2, y2));
                unsafe { bf_queue_free(q) };
            }

            /// Error path: a leading dimension below the row count is rejected with a status and a
            /// message, not undefined behaviour. Positive control: the same call with a valid `lda`
            /// succeeds. (Measured, not assumed: `nrm2` with `incx = 0` is accepted, not rejected;
            /// it reads element 0 `n` times, giving 2.0 for `[1, -2, 3, -4]`. So a zero increment
            /// is not a usable error case, and this test uses `lda` instead.)
            #[test]
            fn bad_leading_dimension_is_an_error_not_a_crash() {
                let q = cpu_queue();
                let a = Usm::new(q, &[1.0 as $t, 2.0, 3.0, 4.0]);
                let x = Usm::new(q, &[1.0 as $t, 1.0]);
                let y = Usm::new(q, &[0.0 as $t, 0.0]);
                assert_eq!(
                    unsafe { $gemv(q, 0, 2, 2, 1.0, a.p, 2, x.p, 1, 0.0, y.p, 1) },
                    0
                );
                let rc = unsafe { $gemv(q, 0, 2, 2, 1.0, a.p, 1, x.p, 1, 0.0, y.p, 1) };
                assert_ne!(rc, 0, "lda = 1 < m = 2 must be rejected");
                assert!(!err().is_empty());
                drop((a, x, y));
                unsafe { bf_queue_free(q) };
            }
        }
    };
}

blas1_tests!(
    single, f32, bf_saxpy, bf_sscal, bf_scopy, bf_sswap, bf_sdot, bf_snrm2, bf_sasum, bf_siamax,
    bf_sgemv
);
blas1_tests!(
    double, f64, bf_daxpy, bf_dscal, bf_dcopy, bf_dswap, bf_ddot, bf_dnrm2, bf_dasum, bf_diamax,
    bf_dgemv
);
