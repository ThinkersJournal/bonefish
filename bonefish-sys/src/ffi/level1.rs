//! Foreign declarations: level 1 BLAS (and `rotm`). Split out of `lib.rs` (R15).

use std::ffi::c_void;

unsafe extern "C" {
    pub fn bf_saxpy(
        q: *mut c_void,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
    ) -> i32;
    pub fn bf_sscal(q: *mut c_void, n: i64, alpha: f32, x: *mut f32, incx: i64) -> i32;
    pub fn bf_scopy(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
    ) -> i32;
    pub fn bf_sswap(q: *mut c_void, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64) -> i32;
    /// Scalar results (`dot`, `nrm2`, `asum`, `iamax`) are written through a USM `result`.
    pub fn bf_sdot(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        result: *mut f32,
    ) -> i32;
    pub fn bf_snrm2(q: *mut c_void, n: i64, x: *const f32, incx: i64, result: *mut f32) -> i32;
    pub fn bf_sasum(q: *mut c_void, n: i64, x: *const f32, incx: i64, result: *mut f32) -> i32;
    /// Zero-based index of the first element of largest magnitude.
    pub fn bf_siamax(q: *mut c_void, n: i64, x: *const f32, incx: i64, result: *mut i64) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_siamin(q: *mut c_void, n: i64, x: *const f32, incx: i64, result: *mut i64) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_srot(
        q: *mut c_void,
        n: i64,
        x: *mut f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        c: f32,
        s: f32,
    ) -> i32;
    /// R2 remainder. `uplo`: 0 upper, 1 lower; `diag`: 0 non-unit, 1 unit; `trans`: 0 N, 1 T, 2 C.
    #[allow(clippy::too_many_arguments)]
    pub fn bf_srotm(
        q: *mut c_void,
        n: i64,
        x: *mut f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        param: *mut f32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_drotm(
        q: *mut c_void,
        n: i64,
        x: *mut f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        param: *mut f64,
    ) -> i32;
    pub fn bf_daxpy(
        q: *mut c_void,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
    ) -> i32;
    pub fn bf_dscal(q: *mut c_void, n: i64, alpha: f64, x: *mut f64, incx: i64) -> i32;
    pub fn bf_dcopy(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
    ) -> i32;
    pub fn bf_dswap(q: *mut c_void, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64) -> i32;
    /// Scalar results (`dot`, `nrm2`, `asum`, `iamax`) are written through a USM `result`.
    pub fn bf_ddot(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        result: *mut f64,
    ) -> i32;
    pub fn bf_dnrm2(q: *mut c_void, n: i64, x: *const f64, incx: i64, result: *mut f64) -> i32;
    pub fn bf_dasum(q: *mut c_void, n: i64, x: *const f64, incx: i64, result: *mut f64) -> i32;
    /// Zero-based index of the first element of largest magnitude.
    pub fn bf_diamax(q: *mut c_void, n: i64, x: *const f64, incx: i64, result: *mut i64) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_diamin(q: *mut c_void, n: i64, x: *const f64, incx: i64, result: *mut i64) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_drot(
        q: *mut c_void,
        n: i64,
        x: *mut f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        c: f64,
        s: f64,
    ) -> i32;
    /// Non-blocking level 1 / `gemv` / DFT (`bf_*_async`): each returns the event, or null on
    /// failure. Same arguments as the blocking function plus `deps`/`n_deps`. The caller keeps
    /// the operands alive, and untouched, until the event or the queue has been waited on;
    /// scalar results are written when the work runs.
    #[allow(clippy::too_many_arguments)]
    pub fn bf_saxpy_async(
        q: *mut c_void,
        n: i64,
        alpha: f32,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sscal_async(
        q: *mut c_void,
        n: i64,
        alpha: f32,
        x: *mut f32,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_scopy_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sswap_async(
        q: *mut c_void,
        n: i64,
        x: *mut f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sdot_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        result: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_snrm2_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_sasum_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_siamax_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_siamin_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_srot_async(
        q: *mut c_void,
        n: i64,
        x: *mut f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        c: f32,
        s: f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_srotm_async(
        q: *mut c_void,
        n: i64,
        x: *mut f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        param: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_drotm_async(
        q: *mut c_void,
        n: i64,
        x: *mut f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        param: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_daxpy_async(
        q: *mut c_void,
        n: i64,
        alpha: f64,
        x: *const f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dscal_async(
        q: *mut c_void,
        n: i64,
        alpha: f64,
        x: *mut f64,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dcopy_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dswap_async(
        q: *mut c_void,
        n: i64,
        x: *mut f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ddot_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        result: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dnrm2_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        result: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_dasum_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        result: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_diamax_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_diamin_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_drot_async(
        q: *mut c_void,
        n: i64,
        x: *mut f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        c: f64,
        s: f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    /// Complex level 1 and `gemv` (`c` single, `z` double), non-blocking. Complex data is
    /// interleaved (re, im); complex scalars are passed by POINTER to two reals; `dotu`/`dotc`
    /// write a complex result through a pointer to two reals; `nrm2`/`asum` of complex data
    /// write a REAL; `iamax` writes a zero-based `i64`. Each returns the event, or null.
    #[allow(clippy::too_many_arguments)]
    pub fn bf_caxpy_async(
        q: *mut c_void,
        n: i64,
        alpha: *const f32,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_cscal_async(
        q: *mut c_void,
        n: i64,
        alpha: *const f32,
        x: *mut f32,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ccopy_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_cswap_async(
        q: *mut c_void,
        n: i64,
        x: *mut f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_cdotu_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        result: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_cdotc_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *const f32,
        incy: i64,
        result: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_cnrm2_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_casum_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut f32,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ciamax_async(
        q: *mut c_void,
        n: i64,
        x: *const f32,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zaxpy_async(
        q: *mut c_void,
        n: i64,
        alpha: *const f64,
        x: *const f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zscal_async(
        q: *mut c_void,
        n: i64,
        alpha: *const f64,
        x: *mut f64,
        incx: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zcopy_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zswap_async(
        q: *mut c_void,
        n: i64,
        x: *mut f64,
        incx: i64,
        y: *mut f64,
        incy: i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zdotu_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        result: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zdotc_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        y: *const f64,
        incy: i64,
        result: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_znrm2_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        result: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_zasum_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        result: *mut f64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
    #[allow(clippy::too_many_arguments)]
    pub fn bf_ziamax_async(
        q: *mut c_void,
        n: i64,
        x: *const f64,
        incx: i64,
        result: *mut i64,
        deps: *const *mut c_void,
        n_deps: i64,
    ) -> *mut c_void;
}
