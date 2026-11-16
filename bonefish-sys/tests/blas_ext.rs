//! R2 remainder: `iamin`, `rot` and `ger` through the shim on the CPU (OpenCL) device, blocking
//! and non-blocking, single and double precision from one macro. Compiled out without oneAPI; the
//! oneAPI CI job sets `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
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
