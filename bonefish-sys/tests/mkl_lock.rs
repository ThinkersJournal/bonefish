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
//!    `src/ffi/*.rs` must be named in exactly one of `MKL_FUNCS` / `NOT_MKL_FUNCS` below, and every
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
    "bf_sgemm_batch_strided_async",
    "bf_dgemm_batch_strided_async",
    "bf_saxpy_async",
    "bf_sscal_async",
    "bf_scopy_async",
    "bf_sswap_async",
    "bf_sdot_async",
    "bf_snrm2_async",
    "bf_sasum_async",
    "bf_siamax_async",
    "bf_sgemv_async",
    "bf_daxpy_async",
    "bf_dscal_async",
    "bf_dcopy_async",
    "bf_dswap_async",
    "bf_ddot_async",
    "bf_dnrm2_async",
    "bf_dasum_async",
    "bf_diamax_async",
    "bf_dgemv_async",
    "bf_dft_compute_async",
    "bf_dft_c2c_new",
    "bf_dft_compute",
    "bf_caxpy_async",
    "bf_cscal_async",
    "bf_ccopy_async",
    "bf_cswap_async",
    "bf_cdotu_async",
    "bf_cdotc_async",
    "bf_cnrm2_async",
    "bf_casum_async",
    "bf_ciamax_async",
    "bf_cgemv_async",
    "bf_zaxpy_async",
    "bf_zscal_async",
    "bf_zcopy_async",
    "bf_zswap_async",
    "bf_zdotu_async",
    "bf_zdotc_async",
    "bf_znrm2_async",
    "bf_zasum_async",
    "bf_ziamax_async",
    "bf_zgemv_async",
    "bf_cgemm",
    "bf_zgemm",
    "bf_cgemm_async",
    "bf_zgemm_async",
    "bf_dftr_new",
    "bf_dftr_free",
    "bf_dftr_compute",
    "bf_dftr_compute_async",
    "bf_dft_free",
    "bf_siamin",
    "bf_siamin_async",
    "bf_srot",
    "bf_srot_async",
    "bf_sger",
    "bf_sger_async",
    "bf_diamin",
    "bf_diamin_async",
    "bf_drot",
    "bf_drot_async",
    "bf_dger",
    "bf_dger_async",
    "bf_srotm",
    "bf_srotm_async",
    "bf_ssymv",
    "bf_ssymv_async",
    "bf_ssyr",
    "bf_ssyr_async",
    "bf_ssyr2",
    "bf_ssyr2_async",
    "bf_strmv",
    "bf_strmv_async",
    "bf_strsv",
    "bf_strsv_async",
    "bf_sgbmv",
    "bf_sgbmv_async",
    "bf_drotm",
    "bf_drotm_async",
    "bf_dsymv",
    "bf_dsymv_async",
    "bf_dsyr",
    "bf_dsyr_async",
    "bf_dsyr2",
    "bf_dsyr2_async",
    "bf_dtrmv",
    "bf_dtrmv_async",
    "bf_dtrsv",
    "bf_dtrsv_async",
    "bf_dgbmv",
    "bf_dgbmv_async",
    "bf_ssymm",
    "bf_ssymm_async",
    "bf_ssyrk",
    "bf_ssyrk_async",
    "bf_ssyr2k",
    "bf_ssyr2k_async",
    "bf_strmm",
    "bf_strmm_async",
    "bf_strsm",
    "bf_strsm_async",
    "bf_dsymm",
    "bf_dsymm_async",
    "bf_dsyrk",
    "bf_dsyrk_async",
    "bf_dsyr2k",
    "bf_dsyr2k_async",
    "bf_dtrmm",
    "bf_dtrmm_async",
    "bf_dtrsm",
    "bf_dtrsm_async",
    "bf_csymm_async",
    "bf_chemm_async",
    "bf_csyrk_async",
    "bf_cherk_async",
    "bf_csyr2k_async",
    "bf_cher2k_async",
    "bf_ctrmm_async",
    "bf_ctrsm_async",
    "bf_zsymm_async",
    "bf_zhemm_async",
    "bf_zsyrk_async",
    "bf_zherk_async",
    "bf_zsyr2k_async",
    "bf_zher2k_async",
    "bf_ztrmm_async",
    "bf_ztrsm_async",
];

/// Functions that do not call oneMKL (SYCL runtime calls and bookkeeping).
const NOT_MKL_FUNCS: &[&str] = &[
    "bf_queue_new",
    "bf_queue_free",
    "bf_queue_device_name",
    "bf_queue_is_in_order",
    "bf_device_count",
    "bf_device_info",
    "bf_queue_new_device",
    "bf_queue_new_ex",
    "bf_event_profiling",
    "bf_noop_async",
    "bf_queue_wait",
    "bf_event_wait",
    "bf_event_free",
    "bf_test_async_failure",
    "bf_test_set_async_chain",
    "bf_malloc",
    "bf_free",
    "bf_memcpy",
    "bf_last_error",
    "bf_mkl_lock_acquisitions",
];

/// The foreign functions `src/ffi/*.rs` declare (split out of `lib.rs` by operation family, R15).
/// The stand-ins used without oneAPI live in `src/stub/` and are written `pub unsafe extern "C"
/// fn`, so only the real declarations (`pub fn bf_...(`) match.
fn declared_ffi_names() -> BTreeSet<String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read src/ffi")
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();
    // Positive control for the enumeration: more than one family file was read.
    assert!(files.len() >= 5, "src/ffi has too few files: {files:?}");
    files
        .iter()
        .flat_map(|path| {
            std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
                .lines()
                .filter_map(|l| l.trim_start().strip_prefix("pub fn bf_"))
                .filter_map(|rest| rest.split('(').next())
                .map(|n| format!("bf_{n}"))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn every_declared_shim_function_is_classified_exactly_once() {
    let declared = declared_ffi_names();
    // Positive control for the parser: it sees functions we know are declared.
    assert!(
        declared.contains("bf_sgemm") && declared.contains("bf_queue_new"),
        "the src/ffi parser found nothing useful: {declared:?}"
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
        "shim functions declared in src/ffi but not classified in tests/mkl_lock.rs: \
         {unclassified:?}. Decide whether each calls oneMKL; if it does it must use \
         bf::guard_mkl / bf::MklLock and go in MKL_FUNCS."
    );
    let stale: Vec<_> = classified.difference(&declared).collect();
    assert!(
        stale.is_empty(),
        "classified here but no longer declared in src/ffi: {stale:?}"
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

    /// One non-blocking call of each level-1 / gemv function for one precision: the lock counter
    /// must move by exactly one for the submission, the event must come back, and it is waited on
    /// and freed before the buffers go.
    macro_rules! async_calls {
        ($q:expr, $t:ty, $axpy:ident, $scal:ident, $copy:ident, $swap:ident, $dot:ident,
         $nrm2:ident, $asum:ident, $iamax:ident, $gemv:ident, $seen:expr) => {{
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
            let none: *const *mut c_void = std::ptr::null();
            macro_rules! one_async {
                ($f:ident, $call:expr) => {{
                    $seen.push(stringify!($f));
                    let (d, ev) = moved(|| unsafe { $call });
                    assert!(!ev.is_null(), "{} returned no event", stringify!($f));
                    assert_eq!(
                        d,
                        1,
                        "{} moved the oneMKL lock counter by {d}, not 1",
                        stringify!($f)
                    );
                    assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{}", stringify!($f));
                    unsafe { bf_event_free(ev) };
                }};
            }
            one_async!($axpy, $axpy($q, 4, 2.0, xp, 1, yp, 1, none, 0));
            one_async!($scal, $scal($q, 4, 2.0, xp, 1, none, 0));
            one_async!($copy, $copy($q, 4, xp, 1, yp, 1, none, 0));
            one_async!($swap, $swap($q, 4, xp, 1, yp, 1, none, 0));
            one_async!($dot, $dot($q, 4, xp, 1, yp, 1, rp, none, 0));
            one_async!($nrm2, $nrm2($q, 4, xp, 1, rp, none, 0));
            one_async!($asum, $asum($q, 4, xp, 1, rp, none, 0));
            one_async!($iamax, $iamax($q, 4, xp, 1, ip, none, 0));
            one_async!(
                $gemv,
                $gemv($q, 0, 2, 2, 1.0, mp, 2, xp, 1, 0.0, yp, 1, none, 0)
            );
        }};
    }

    /// One non-blocking call of each complex level-1 function and `gemv` for one precision: the
    /// lock counter must move by exactly one, an event must come back, and it is waited on and
    /// freed before the buffers go. Complex scalars go in by pointer to two reals.
    macro_rules! complex_level1_calls {
        ($q:expr, $r:ty, $axpy:ident, $scal:ident, $copy:ident, $swap:ident, $dotu:ident,
         $dotc:ident, $nrm2:ident, $asum:ident, $iamax:ident, $gemv:ident, $seen:expr) => {{
            let sz = std::mem::size_of::<$r>() as i64;
            let (x, y, m) = (
                Usm::new($q, 64 * sz),
                Usm::new($q, 64 * sz),
                Usm::new($q, 64 * sz),
            );
            let (cres, rres) = (Usm::new($q, 2 * sz), Usm::new($q, sz));
            let idx = Usm::new($q, 8);
            let (xp, yp, mp) = (x.p.cast::<$r>(), y.p.cast::<$r>(), m.p.cast::<$r>());
            let (cp, rp, ip) = (
                cres.p.cast::<$r>(),
                rres.p.cast::<$r>(),
                idx.p.cast::<i64>(),
            );
            let one: [$r; 2] = [1.0, 0.0];
            let zero: [$r; 2] = [0.0, 0.0];
            let none: *const *mut c_void = std::ptr::null();
            macro_rules! one_cx {
                ($f:ident, $call:expr) => {{
                    $seen.push(stringify!($f));
                    let (d, ev) = moved(|| unsafe { $call });
                    assert!(!ev.is_null(), "{} returned no event", stringify!($f));
                    assert_eq!(
                        d,
                        1,
                        "{} moved the oneMKL lock counter by {d}, not 1",
                        stringify!($f)
                    );
                    assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{}", stringify!($f));
                    unsafe { bf_event_free(ev) };
                }};
            }
            one_cx!($axpy, $axpy($q, 4, one.as_ptr(), xp, 1, yp, 1, none, 0));
            one_cx!($scal, $scal($q, 4, one.as_ptr(), xp, 1, none, 0));
            one_cx!($copy, $copy($q, 4, xp, 1, yp, 1, none, 0));
            one_cx!($swap, $swap($q, 4, xp, 1, yp, 1, none, 0));
            one_cx!($dotu, $dotu($q, 4, xp, 1, yp, 1, cp, none, 0));
            one_cx!($dotc, $dotc($q, 4, xp, 1, yp, 1, cp, none, 0));
            one_cx!($nrm2, $nrm2($q, 4, xp, 1, rp, none, 0));
            one_cx!($asum, $asum($q, 4, xp, 1, rp, none, 0));
            one_cx!($iamax, $iamax($q, 4, xp, 1, ip, none, 0));
            one_cx!(
                $gemv,
                $gemv(
                    $q,
                    0,
                    2,
                    2,
                    one.as_ptr(),
                    mp,
                    2,
                    xp,
                    1,
                    zero.as_ptr(),
                    yp,
                    1,
                    none,
                    0
                )
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
            let (d, e) = moved(|| unsafe {
                bf_sgemm_batch_strided_async(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    1.0,
                    m32.p.cast(),
                    2,
                    4,
                    m32.p.cast(),
                    2,
                    4,
                    0.0,
                    c32.p.cast(),
                    2,
                    4,
                    1,
                    std::ptr::null(),
                    0,
                )
            });
            seen.push("bf_sgemm_batch_strided_async");
            assert!(!e.is_null());
            assert_eq!(
                d, 1,
                "bf_sgemm_batch_strided_async moved the counter by {d}, not 1"
            );
            assert_eq!(unsafe { bf_event_wait(e) }, 0);
            unsafe { bf_event_free(e) };
            let (d, e) = moved(|| unsafe {
                bf_dgemm_batch_strided_async(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    1.0,
                    m64.p.cast(),
                    2,
                    4,
                    m64.p.cast(),
                    2,
                    4,
                    0.0,
                    c64.p.cast(),
                    2,
                    4,
                    1,
                    std::ptr::null(),
                    0,
                )
            });
            seen.push("bf_dgemm_batch_strided_async");
            assert!(!e.is_null());
            assert_eq!(
                d, 1,
                "bf_dgemm_batch_strided_async moved the counter by {d}, not 1"
            );
            assert_eq!(unsafe { bf_event_wait(e) }, 0);
            unsafe { bf_event_free(e) };
        }

        async_calls!(
            q,
            f32,
            bf_saxpy_async,
            bf_sscal_async,
            bf_scopy_async,
            bf_sswap_async,
            bf_sdot_async,
            bf_snrm2_async,
            bf_sasum_async,
            bf_siamax_async,
            bf_sgemv_async,
            seen
        );
        async_calls!(
            q,
            f64,
            bf_daxpy_async,
            bf_dscal_async,
            bf_dcopy_async,
            bf_dswap_async,
            bf_ddot_async,
            bf_dnrm2_async,
            bf_dasum_async,
            bf_diamax_async,
            bf_dgemv_async,
            seen
        );

        // iamin, rot, ger: blocking and non-blocking, both precisions, one lock entry each.
        macro_rules! ext_calls {
            ($t:ty, $iamin:ident, $rot:ident, $ger:ident, $iamin_a:ident, $rot_a:ident,
             $ger_a:ident) => {{
                let sz = std::mem::size_of::<$t>() as i64;
                let (x, y, m) = (
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                );
                let (xp, yp, mp) = (x.p.cast::<$t>(), y.p.cast::<$t>(), m.p.cast::<$t>());
                let idx = Usm::new(q, 8);
                let ip = idx.p.cast::<i64>();
                let none: *const *mut c_void = std::ptr::null();
                seen.push(stringify!($iamin));
                let (d, rc) = moved(|| unsafe { $iamin(q, 4, xp, 1, ip) });
                assert_eq!((rc, d), (0, 1), "{}", stringify!($iamin));
                seen.push(stringify!($rot));
                let (d, rc) = moved(|| unsafe { $rot(q, 4, xp, 1, yp, 1, 1.0, 0.0) });
                assert_eq!((rc, d), (0, 1), "{}", stringify!($rot));
                seen.push(stringify!($ger));
                let (d, rc) = moved(|| unsafe { $ger(q, 2, 2, 1.0, xp, 1, yp, 1, mp, 2) });
                assert_eq!((rc, d), (0, 1), "{}", stringify!($ger));
                macro_rules! one_ext {
                    ($f:ident, $call:expr) => {{
                        seen.push(stringify!($f));
                        let (d, ev) = moved(|| unsafe { $call });
                        assert!(!ev.is_null(), "{} returned no event", stringify!($f));
                        assert_eq!(d, 1, "{} moved the counter by {d}, not 1", stringify!($f));
                        assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{}", stringify!($f));
                        unsafe { bf_event_free(ev) };
                    }};
                }
                one_ext!($iamin_a, $iamin_a(q, 4, xp, 1, ip, none, 0));
                one_ext!($rot_a, $rot_a(q, 4, xp, 1, yp, 1, 1.0, 0.0, none, 0));
                one_ext!($ger_a, $ger_a(q, 2, 2, 1.0, xp, 1, yp, 1, mp, 2, none, 0));
            }};
        }
        ext_calls!(
            f32,
            bf_siamin,
            bf_srot,
            bf_sger,
            bf_siamin_async,
            bf_srot_async,
            bf_sger_async
        );
        ext_calls!(
            f64,
            bf_diamin,
            bf_drot,
            bf_dger,
            bf_diamin_async,
            bf_drot_async,
            bf_dger_async
        );

        // rotm, symv, syr, syr2, trmv, trsv, gbmv: blocking and non-blocking, both precisions.
        macro_rules! l2_calls {
            ($t:ty, $rotm:ident, $symv:ident, $syr:ident, $syr2:ident, $trmv:ident, $trsv:ident, $gbmv:ident, $rotm_a:ident, $symv_a:ident, $syr_a:ident, $syr2_a:ident, $trmv_a:ident, $trsv_a:ident, $gbmv_a:ident) => {{
                let sz = std::mem::size_of::<$t>() as i64;
                let (x, y, m) = (
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                );
                let (xp, yp, mp) = (x.p.cast::<$t>(), y.p.cast::<$t>(), m.p.cast::<$t>());
                let none: *const *mut c_void = std::ptr::null();
                macro_rules! pair {
                    ($blk:ident, $asy:ident, $bcall:expr, $acall:expr) => {{
                        seen.push(stringify!($blk));
                        let (d, rc) = moved(|| unsafe { $bcall });
                        assert_eq!((rc, d), (0, 1), "{}", stringify!($blk));
                        seen.push(stringify!($asy));
                        let (d, ev) = moved(|| unsafe { $acall });
                        assert!(!ev.is_null(), "{} returned no event", stringify!($asy));
                        assert_eq!(d, 1, "{} moved the counter by {d}, not 1", stringify!($asy));
                        assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{}", stringify!($asy));
                        unsafe { bf_event_free(ev) };
                    }};
                }
                pair!(
                    $rotm,
                    $rotm_a,
                    $rotm(q, 2, xp, 1, yp, 1, mp),
                    $rotm_a(q, 2, xp, 1, yp, 1, mp, none, 0)
                );
                pair!(
                    $symv,
                    $symv_a,
                    $symv(q, 0, 2, 1.0, mp, 2, xp, 1, 0.0, yp, 1),
                    $symv_a(q, 0, 2, 1.0, mp, 2, xp, 1, 0.0, yp, 1, none, 0)
                );
                pair!(
                    $syr,
                    $syr_a,
                    $syr(q, 0, 2, 1.0, xp, 1, mp, 2),
                    $syr_a(q, 0, 2, 1.0, xp, 1, mp, 2, none, 0)
                );
                pair!(
                    $syr2,
                    $syr2_a,
                    $syr2(q, 0, 2, 1.0, xp, 1, yp, 1, mp, 2),
                    $syr2_a(q, 0, 2, 1.0, xp, 1, yp, 1, mp, 2, none, 0)
                );
                pair!(
                    $trmv,
                    $trmv_a,
                    $trmv(q, 0, 0, 0, 2, mp, 2, xp, 1),
                    $trmv_a(q, 0, 0, 0, 2, mp, 2, xp, 1, none, 0)
                );
                pair!(
                    $trsv,
                    $trsv_a,
                    $trsv(q, 0, 0, 0, 2, mp, 2, xp, 1),
                    $trsv_a(q, 0, 0, 0, 2, mp, 2, xp, 1, none, 0)
                );
                pair!(
                    $gbmv,
                    $gbmv_a,
                    $gbmv(q, 0, 2, 2, 0, 0, 1.0, mp, 1, xp, 1, 0.0, yp, 1),
                    $gbmv_a(q, 0, 2, 2, 0, 0, 1.0, mp, 1, xp, 1, 0.0, yp, 1, none, 0)
                );
            }};
        }
        l2_calls!(
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
        l2_calls!(
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

        // symm, syrk, syr2k, trmm, trsm: blocking and non-blocking, both precisions.
        macro_rules! l3_calls {
            ($t:ty, $xp:ident, $yp:ident, $mp:ident, $($blk:ident, $asy:ident, ($($arg:expr),*), ($($arg_a:expr),*));+ $(;)?) => {{
                let sz = std::mem::size_of::<$t>() as i64;
                let (x, y, m) = (
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                );
                let ($xp, $yp, $mp) = (x.p.cast::<$t>(), y.p.cast::<$t>(), m.p.cast::<$t>());
                let none: *const *mut c_void = std::ptr::null();
                $(
                    seen.push(stringify!($blk));
                    let (d, rc) = moved(|| unsafe { $blk(q, $($arg),*) });
                    assert_eq!((rc, d), (0, 1), "{}", stringify!($blk));
                    seen.push(stringify!($asy));
                    let (d, ev) = moved(|| unsafe { $asy(q, $($arg_a),*, none, 0) });
                    assert!(!ev.is_null(), "{} returned no event", stringify!($asy));
                    assert_eq!(d, 1, "{} moved the counter by {d}, not 1", stringify!($asy));
                    assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{}", stringify!($asy));
                    unsafe { bf_event_free(ev) };
                )+
            }};
        }
        l3_calls!(f32, xp, yp, mp,
            bf_ssymm, bf_ssymm_async, (0, 0, 2, 2, 1.0, mp, 2, xp, 2, 0.0, yp, 2), (0, 0, 2, 2, 1.0, mp, 2, xp, 2, 0.0, yp, 2);
            bf_ssyrk, bf_ssyrk_async, (0, 0, 2, 1, 1.0, mp, 2, 0.0, yp, 2), (0, 0, 2, 1, 1.0, mp, 2, 0.0, yp, 2);
            bf_ssyr2k, bf_ssyr2k_async, (0, 0, 2, 1, 1.0, mp, 2, xp, 2, 0.0, yp, 2), (0, 0, 2, 1, 1.0, mp, 2, xp, 2, 0.0, yp, 2);
            bf_strmm, bf_strmm_async, (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2), (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2);
            bf_strsm, bf_strsm_async, (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2), (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2)
        );
        l3_calls!(f64, xp, yp, mp,
            bf_dsymm, bf_dsymm_async, (0, 0, 2, 2, 1.0, mp, 2, xp, 2, 0.0, yp, 2), (0, 0, 2, 2, 1.0, mp, 2, xp, 2, 0.0, yp, 2);
            bf_dsyrk, bf_dsyrk_async, (0, 0, 2, 1, 1.0, mp, 2, 0.0, yp, 2), (0, 0, 2, 1, 1.0, mp, 2, 0.0, yp, 2);
            bf_dsyr2k, bf_dsyr2k_async, (0, 0, 2, 1, 1.0, mp, 2, xp, 2, 0.0, yp, 2), (0, 0, 2, 1, 1.0, mp, 2, xp, 2, 0.0, yp, 2);
            bf_dtrmm, bf_dtrmm_async, (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2), (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2);
            bf_dtrsm, bf_dtrsm_async, (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2), (0, 0, 0, 0, 2, 2, 1.0, mp, 2, xp, 2)
        );

        // Complex level 3 (non-blocking only): c and z symm, hemm, syrk, herk, syr2k, her2k, trmm, trsm.
        macro_rules! cl3_calls {
            ($t:ty, $xp:ident, $yp:ident, $mp:ident, $one:ident, $zero:ident, $($f:ident, ($($arg:expr),*));+ $(;)?) => {{
                let sz = std::mem::size_of::<$t>() as i64;
                let (x, y, m) = (
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                    Usm::new(q, 16 * sz),
                );
                let ($xp, $yp, $mp) = (x.p.cast::<$t>(), y.p.cast::<$t>(), m.p.cast::<$t>());
                let one_arr: [$t; 2] = [1.0, 0.0];
                let zero_arr: [$t; 2] = [0.0, 0.0];
                let ($one, $zero) = (one_arr.as_ptr(), zero_arr.as_ptr());
                let none: *const *mut c_void = std::ptr::null();
                $(
                    seen.push(stringify!($f));
                    let (d, ev) = moved(|| unsafe { $f(q, $($arg),*, none, 0) });
                    assert!(!ev.is_null(), "{} returned no event", stringify!($f));
                    assert_eq!(d, 1, "{} moved the counter by {d}, not 1", stringify!($f));
                    assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{}", stringify!($f));
                    unsafe { bf_event_free(ev) };
                )+
            }};
        }
        cl3_calls!(f32, xp, yp, mp, one, zero,
            bf_csymm_async, (0, 0, 2, 2, one, mp, 2, xp, 2, zero, yp, 2);
            bf_chemm_async, (0, 0, 2, 2, one, mp, 2, xp, 2, zero, yp, 2);
            bf_csyrk_async, (0, 0, 2, 1, one, mp, 2, zero, yp, 2);
            bf_cherk_async, (0, 0, 2, 1, 1.0, mp, 2, 0.0, yp, 2);
            bf_csyr2k_async, (0, 0, 2, 1, one, mp, 2, xp, 2, zero, yp, 2);
            bf_cher2k_async, (0, 0, 2, 1, one, mp, 2, xp, 2, 0.0, yp, 2);
            bf_ctrmm_async, (0, 0, 0, 0, 2, 2, one, mp, 2, xp, 2);
            bf_ctrsm_async, (0, 0, 0, 0, 2, 2, one, mp, 2, xp, 2)
        );
        cl3_calls!(f64, xp, yp, mp, one, zero,
            bf_zsymm_async, (0, 0, 2, 2, one, mp, 2, xp, 2, zero, yp, 2);
            bf_zhemm_async, (0, 0, 2, 2, one, mp, 2, xp, 2, zero, yp, 2);
            bf_zsyrk_async, (0, 0, 2, 1, one, mp, 2, zero, yp, 2);
            bf_zherk_async, (0, 0, 2, 1, 1.0, mp, 2, 0.0, yp, 2);
            bf_zsyr2k_async, (0, 0, 2, 1, one, mp, 2, xp, 2, zero, yp, 2);
            bf_zher2k_async, (0, 0, 2, 1, one, mp, 2, xp, 2, 0.0, yp, 2);
            bf_ztrmm_async, (0, 0, 0, 0, 2, 2, one, mp, 2, xp, 2);
            bf_ztrsm_async, (0, 0, 0, 0, 2, 2, one, mp, 2, xp, 2)
        );

        // Non-blocking DFT compute: a plan, one async compute, wait, free.
        {
            let dims = [4_i64];
            let h = unsafe { bf_dft_c2c_new(q, 1, 1, dims.as_ptr(), 1, 1.0, 1.0) };
            assert!(!h.is_null());
            let data = Usm::new(q, 4 * 16);
            let (d, ev) = moved(|| unsafe {
                bf_dft_compute_async(h, 0, data.p, std::ptr::null_mut(), std::ptr::null(), 0)
            });
            seen.push("bf_dft_compute_async");
            assert!(!ev.is_null(), "bf_dft_compute_async returned no event");
            assert_eq!(d, 1, "bf_dft_compute_async moved the counter by {d}, not 1");
            assert_eq!(unsafe { bf_event_wait(ev) }, 0);
            unsafe { bf_event_free(ev) };
            unsafe { bf_dft_free(h) };
            drop(data);
        }

        // Complex gemm, single and double, blocking and non-blocking: 2 x 2 matrices of complex
        // numbers (2 reals each), alpha and beta passed by pointer.
        {
            let (a32, c32) = (Usm::new(q, 4 * 8), Usm::new(q, 4 * 8));
            let (a64, c64) = (Usm::new(q, 4 * 16), Usm::new(q, 4 * 16));
            let (one32, zero32) = ([1.0_f32, 0.0], [0.0_f32, 0.0]);
            let (one64, zero64) = ([1.0_f64, 0.0], [0.0_f64, 0.0]);
            let (d, rc) = moved(|| unsafe {
                bf_cgemm(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    one32.as_ptr(),
                    a32.p.cast(),
                    2,
                    a32.p.cast(),
                    2,
                    zero32.as_ptr(),
                    c32.p.cast(),
                    2,
                )
            });
            seen.push("bf_cgemm");
            assert_eq!((rc, d), (0, 1), "bf_cgemm: status {rc}, counter moved {d}");
            let (d, rc) = moved(|| unsafe {
                bf_zgemm(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    one64.as_ptr(),
                    a64.p.cast(),
                    2,
                    a64.p.cast(),
                    2,
                    zero64.as_ptr(),
                    c64.p.cast(),
                    2,
                )
            });
            seen.push("bf_zgemm");
            assert_eq!((rc, d), (0, 1), "bf_zgemm: status {rc}, counter moved {d}");
            let (d, ev) = moved(|| unsafe {
                bf_cgemm_async(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    one32.as_ptr(),
                    a32.p.cast(),
                    2,
                    a32.p.cast(),
                    2,
                    zero32.as_ptr(),
                    c32.p.cast(),
                    2,
                    std::ptr::null(),
                    0,
                )
            });
            seen.push("bf_cgemm_async");
            assert!(!ev.is_null(), "bf_cgemm_async returned no event");
            assert_eq!(d, 1, "bf_cgemm_async moved the counter by {d}, not 1");
            assert_eq!(unsafe { bf_event_wait(ev) }, 0);
            unsafe { bf_event_free(ev) };
            let (d, ev) = moved(|| unsafe {
                bf_zgemm_async(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    one64.as_ptr(),
                    a64.p.cast(),
                    2,
                    a64.p.cast(),
                    2,
                    zero64.as_ptr(),
                    c64.p.cast(),
                    2,
                    std::ptr::null(),
                    0,
                )
            });
            seen.push("bf_zgemm_async");
            assert!(!ev.is_null(), "bf_zgemm_async returned no event");
            assert_eq!(d, 1, "bf_zgemm_async moved the counter by {d}, not 1");
            assert_eq!(unsafe { bf_event_wait(ev) }, 0);
            unsafe { bf_event_free(ev) };
            drop((a32, c32, a64, c64));
        }

        complex_level1_calls!(
            q,
            f32,
            bf_caxpy_async,
            bf_cscal_async,
            bf_ccopy_async,
            bf_cswap_async,
            bf_cdotu_async,
            bf_cdotc_async,
            bf_cnrm2_async,
            bf_casum_async,
            bf_ciamax_async,
            bf_cgemv_async,
            seen
        );
        complex_level1_calls!(
            q,
            f64,
            bf_zaxpy_async,
            bf_zscal_async,
            bf_zcopy_async,
            bf_zswap_async,
            bf_zdotu_async,
            bf_zdotc_async,
            bf_znrm2_async,
            bf_zasum_async,
            bf_ziamax_async,
            bf_zgemv_async,
            seen
        );

        // Real DFT: create, blocking compute, async compute, free are four separate entries. Length
        // 4 real -> 3 complex (4/2 + 1) in the half spectrum.
        {
            let dims = [4_i64];
            let (d, h) = moved(|| unsafe { bf_dftr_new(q, 1, 1, dims.as_ptr(), 1.0, 1.0) });
            seen.push("bf_dftr_new");
            assert!(!h.is_null(), "bf_dftr_new failed");
            assert_eq!(d, 1, "bf_dftr_new moved the counter by {d}, not 1");
            let real = Usm::new(q, 4 * 8);
            let half = Usm::new(q, 3 * 16);
            let (d, rc) = moved(|| unsafe { bf_dftr_compute(h, 0, real.p, half.p) });
            seen.push("bf_dftr_compute");
            assert_eq!(
                (rc, d),
                (0, 1),
                "bf_dftr_compute: status {rc}, counter moved {d}"
            );
            let (d, ev) = moved(|| unsafe {
                bf_dftr_compute_async(h, 0, real.p, half.p, std::ptr::null(), 0)
            });
            seen.push("bf_dftr_compute_async");
            assert!(!ev.is_null(), "bf_dftr_compute_async returned no event");
            assert_eq!(
                d, 1,
                "bf_dftr_compute_async moved the counter by {d}, not 1"
            );
            assert_eq!(unsafe { bf_event_wait(ev) }, 0);
            unsafe { bf_event_free(ev) };
            let (d, ()) = moved(|| unsafe { bf_dftr_free(h) });
            seen.push("bf_dftr_free");
            assert_eq!(d, 1, "bf_dftr_free moved the counter by {d}, not 1");
            drop((real, half));
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
