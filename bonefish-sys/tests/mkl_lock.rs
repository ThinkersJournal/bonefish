//! Every shim function that calls oneMKL must go through the process-wide oneMKL lock.
//!
//! Why this is tested: the shipped oneMKL is not host-thread-safe (see `ROADMAP.md` R3b and
//! `docs/INTEL-REPORT-DRAFT.md`), and `Queue`/`UsmBuffer` are `Send + Sync` only because the shim
//! serialises those calls. One future function that calls oneMKL without the lock would quietly
//! void that, and no ordinary test would notice, since a single thread cannot tell.
//!
//! How it is checked, in two parts:
//!
//! 1. **Classification (no device needed, always runs).** Every foreign function declared in
//!    `src/lib.rs` must be named in exactly one of `MKL_FUNCS` / `NOT_MKL_FUNCS` below, and every
//!    name below must still be declared. Adding a shim function without classifying it fails here,
//!    so it cannot go unexamined.
//! 2. **Counter (needs a device).** The shim exports `bf_mkl_lock_acquisitions`, which counts
//!    entries into the lock path. Each function classified as oneMKL is called once and must move
//!    the counter by exactly one; each function classified as NOT oneMKL must leave it unchanged,
//!    which is the control showing the counter is not simply always moving.
//!
//! What this does not catch: a function that is *classified* as not-oneMKL while really calling
//! oneMKL. That is a review question, which is why the classification is a short explicit list.

use std::collections::BTreeSet;

/// Functions that call oneMKL and must take the lock.
const MKL_FUNCS: &[&str] = &[
    "bf_sgemm",
    "bf_dgemm",
    "bf_saxpy",
    "bf_sscal",
    "bf_scopy",
    "bf_sswap",
    "bf_sdot",
    "bf_snrm2",
    "bf_sasum",
    "bf_siamax",
    "bf_sgemv",
    "bf_daxpy",
    "bf_dscal",
    "bf_dcopy",
    "bf_dswap",
    "bf_ddot",
    "bf_dnrm2",
    "bf_dasum",
    "bf_diamax",
    "bf_dgemv",
    "bf_sgemm_async",
    "bf_dgemm_async",
    "bf_dft_c2c_new",
    "bf_dft_compute",
    "bf_dft_free",
];

/// Functions that do not call oneMKL (SYCL runtime calls and bookkeeping).
const NOT_MKL_FUNCS: &[&str] = &[
    "bf_queue_new",
    "bf_queue_free",
    "bf_queue_device_name",
    "bf_queue_is_in_order",
    "bf_queue_wait",
    "bf_event_wait",
    "bf_event_free",
    "bf_test_async_failure",
    "bf_malloc",
    "bf_free",
    "bf_memcpy",
    "bf_last_error",
    "bf_mkl_lock_acquisitions",
];

/// The foreign functions `src/lib.rs` declares. The stand-ins used without oneAPI are written
/// `pub unsafe extern "C" fn`, so only the real declarations (`pub fn bf_...(`) match.
fn declared_ffi_names() -> BTreeSet<String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs");
    let src = std::fs::read_to_string(path).expect("read src/lib.rs");
    src.lines()
        .filter_map(|l| l.trim_start().strip_prefix("pub fn bf_"))
        .filter_map(|rest| rest.split('(').next())
        .map(|n| format!("bf_{n}"))
        .collect()
}

#[test]
fn every_declared_shim_function_is_classified_exactly_once() {
    let declared = declared_ffi_names();
    // Positive control for the parser: it sees functions we know are declared.
    assert!(
        declared.contains("bf_sgemm") && declared.contains("bf_queue_new"),
        "the lib.rs parser found nothing useful: {declared:?}"
    );
    let mkl: BTreeSet<String> = MKL_FUNCS.iter().map(|s| s.to_string()).collect();
    let not_mkl: BTreeSet<String> = NOT_MKL_FUNCS.iter().map(|s| s.to_string()).collect();
    assert!(
        mkl.is_disjoint(&not_mkl),
        "classified both ways: {:?}",
        mkl.intersection(&not_mkl).collect::<Vec<_>>()
    );
    let classified: BTreeSet<String> = mkl.union(&not_mkl).cloned().collect();
    let unclassified: Vec<_> = declared.difference(&classified).collect();
    assert!(
        unclassified.is_empty(),
        "shim functions declared in src/lib.rs but not classified in tests/mkl_lock.rs: \
         {unclassified:?}. Decide whether each calls oneMKL; if it does it must use \
         bf::guard_mkl / bf::MklLock and go in MKL_FUNCS."
    );
    let stale: Vec<_> = classified.difference(&declared).collect();
    assert!(
        stale.is_empty(),
        "classified here but no longer declared in src/lib.rs: {stale:?}"
    );
}

#[cfg(bonefish_sycl)]
mod counter {
    use super::*;
    use bonefish_sys::*;
    use std::ffi::c_void;

    fn acquisitions() -> i64 {
        unsafe { bf_mkl_lock_acquisitions() }
    }

    /// Run `f`, return how far the lock counter moved and what `f` returned.
    fn moved<R>(f: impl FnOnce() -> R) -> (i64, R) {
        let before = acquisitions();
        let r = f();
        (acquisitions() - before, r)
    }

    struct Usm {
        q: *mut c_void,
        p: *mut c_void,
    }
    impl Usm {
        fn new(q: *mut c_void, bytes: i64) -> Self {
            let p = unsafe { bf_malloc(q, 0, bytes) };
            assert!(!p.is_null());
            unsafe { std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize) };
            Usm { q, p }
        }
    }
    impl Drop for Usm {
        fn drop(&mut self) {
            unsafe { bf_free(self.q, self.p) }
        }
    }

    /// One valid call of each real-valued oneMKL function for one precision. Returns the names it
    /// exercised so the caller can check the list against `MKL_FUNCS`.
    macro_rules! calls {
        ($q:expr, $t:ty, $axpy:ident, $scal:ident, $copy:ident, $swap:ident, $dot:ident,
         $nrm2:ident, $asum:ident, $iamax:ident, $gemv:ident, $gemm:ident, $seen:expr) => {{
            let sz = std::mem::size_of::<$t>() as i64;
            let (x, y, r, m) = (
                Usm::new($q, 16 * sz),
                Usm::new($q, 16 * sz),
                Usm::new($q, sz),
                Usm::new($q, 16 * sz),
            );
            let (xp, yp, rp, mp) = (
                x.p.cast::<$t>(),
                y.p.cast::<$t>(),
                r.p.cast::<$t>(),
                m.p.cast::<$t>(),
            );
            let idx = Usm::new($q, 8);
            let ip = idx.p.cast::<i64>();
            macro_rules! one {
                ($f:ident, $call:expr) => {{
                    $seen.push(stringify!($f));
                    let (d, rc) = moved(|| unsafe { $call });
                    assert_eq!(rc, 0, "{} failed", stringify!($f));
                    assert_eq!(
                        d,
                        1,
                        "{} moved the oneMKL lock counter by {d}, not 1",
                        stringify!($f)
                    );
                }};
            }
            one!($axpy, $axpy($q, 4, 2.0, xp, 1, yp, 1));
            one!($scal, $scal($q, 4, 2.0, xp, 1));
            one!($copy, $copy($q, 4, xp, 1, yp, 1));
            one!($swap, $swap($q, 4, xp, 1, yp, 1));
            one!($dot, $dot($q, 4, xp, 1, yp, 1, rp));
            one!($nrm2, $nrm2($q, 4, xp, 1, rp));
            one!($asum, $asum($q, 4, xp, 1, rp));
            one!($iamax, $iamax($q, 4, xp, 1, ip));
            one!($gemv, $gemv($q, 0, 2, 2, 1.0, mp, 2, xp, 1, 0.0, yp, 1));
            one!(
                $gemm,
                $gemm($q, 0, 0, 2, 2, 2, 1.0, mp, 2, mp, 2, 0.0, yp, 2)
            );
        }};
    }

    #[test]
    fn every_onemkl_function_takes_the_lock_exactly_once() {
        let q = unsafe { bf_queue_new(1) };
        assert!(!q.is_null(), "no CPU SYCL device");
        let mut seen: Vec<&'static str> = Vec::new();

        calls!(
            q, f32, bf_saxpy, bf_sscal, bf_scopy, bf_sswap, bf_sdot, bf_snrm2, bf_sasum, bf_siamax,
            bf_sgemv, bf_sgemm, seen
        );
        calls!(
            q, f64, bf_daxpy, bf_dscal, bf_dcopy, bf_dswap, bf_ddot, bf_dnrm2, bf_dasum, bf_diamax,
            bf_dgemv, bf_dgemm, seen
        );

        // DFT: create, compute, free are three separate entries.
        let dims = [4_i64];
        let (d, h) = moved(|| unsafe { bf_dft_c2c_new(q, 1, 1, dims.as_ptr(), 1, 1.0, 1.0) });
        seen.push("bf_dft_c2c_new");
        assert!(!h.is_null());
        assert_eq!(d, 1, "bf_dft_c2c_new moved the counter by {d}, not 1");
        let data = Usm::new(q, 4 * 16);
        let (d, rc) = moved(|| unsafe { bf_dft_compute(h, 0, data.p, std::ptr::null_mut()) });
        seen.push("bf_dft_compute");
        assert_eq!(
            (rc, d),
            (0, 1),
            "bf_dft_compute: status {rc}, counter moved {d}"
        );
        let (d, ()) = moved(|| unsafe { bf_dft_free(h) });
        seen.push("bf_dft_free");
        assert_eq!(d, 1, "bf_dft_free moved the counter by {d}, not 1");
        // `data` frees its USM through the queue, so it must go before the queue does. Left to
        // drop at the end of the function it ran after `bf_queue_free(q)`: a use after free that
        // Windows tolerated and the Linux CI run turned into SIGSEGV.
        drop(data);

        // Non-blocking gemm: the lock covers the submission. Wait before the buffers are freed.
        {
            let m32 = Usm::new(q, 16);
            let m64 = Usm::new(q, 32);
            let c32 = Usm::new(q, 16);
            let c64 = Usm::new(q, 32);
            let (d, e) = moved(|| unsafe {
                bf_sgemm_async(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    1.0,
                    m32.p.cast(),
                    2,
                    m32.p.cast(),
                    2,
                    0.0,
                    c32.p.cast(),
                    2,
                    std::ptr::null(),
                    0,
                )
            });
            seen.push("bf_sgemm_async");
            assert!(!e.is_null());
            assert_eq!(d, 1, "bf_sgemm_async moved the counter by {d}, not 1");
            assert_eq!(unsafe { bf_event_wait(e) }, 0);
            unsafe { bf_event_free(e) };
            let (d, e) = moved(|| unsafe {
                bf_dgemm_async(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    1.0,
                    m64.p.cast(),
                    2,
                    m64.p.cast(),
                    2,
                    0.0,
                    c64.p.cast(),
                    2,
                    std::ptr::null(),
                    0,
                )
            });
            seen.push("bf_dgemm_async");
            assert!(!e.is_null());
            assert_eq!(d, 1, "bf_dgemm_async moved the counter by {d}, not 1");
            assert_eq!(unsafe { bf_event_wait(e) }, 0);
            unsafe { bf_event_free(e) };
        }

        // The exercised list must be exactly the classified list (no function silently skipped).
        let exercised: BTreeSet<&str> = seen.iter().copied().collect();
        let classified: BTreeSet<&str> = MKL_FUNCS.iter().copied().collect();
        assert_eq!(exercised, classified, "this test and MKL_FUNCS disagree");
        assert_eq!(
            seen.len(),
            MKL_FUNCS.len(),
            "a function was exercised twice"
        );

        // Control: the functions classified as NOT oneMKL must not move the counter, so the
        // counter is not simply always moving.
        let (d, qq) = moved(|| unsafe { bf_queue_new(1) });
        assert_eq!(d, 0, "bf_queue_new moved the counter");
        let (d, p) = moved(|| unsafe { bf_malloc(qq, 0, 64) });
        assert_eq!(d, 0, "bf_malloc moved the counter");
        let host = [0u8; 64];
        let (d, rc) = moved(|| unsafe { bf_memcpy(qq, p, host.as_ptr().cast(), 64) });
        assert_eq!((rc, d), (0, 0), "bf_memcpy: status {rc}, counter moved {d}");
        let mut name = [0 as std::ffi::c_char; 64];
        let (d, n) = moved(|| unsafe { bf_queue_device_name(qq, name.as_mut_ptr(), 64) });
        assert!(n > 0);
        assert_eq!(d, 0, "bf_queue_device_name moved the counter");
        let (d, ()) = moved(|| unsafe { bf_free(qq, p) });
        assert_eq!(d, 0, "bf_free moved the counter");
        let (d, ()) = moved(|| unsafe { bf_queue_free(qq) });
        assert_eq!(d, 0, "bf_queue_free moved the counter");
        unsafe { bf_queue_free(q) };
    }
}
