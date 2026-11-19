//! Stand-ins for: level 1 BLAS (and `rotm`) (build without the shim). Split out of `lib.rs` (R15).

use super::base::STATUS_UNAVAILABLE;
use std::ffi::c_void;

pub unsafe extern "C" fn bf_saxpy(
    _q: *mut c_void,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_sscal(
    _q: *mut c_void,
    _n: i64,
    _alpha: f32,
    _x: *mut f32,
    _incx: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_scopy(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_sswap(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_sdot(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _result: *mut f32,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_snrm2(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut f32,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_sasum(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut f32,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_siamax(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_siamin(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_srot(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _c: f32,
    _s: f32,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_srotm(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _param: *mut f32,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_drotm(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _param: *mut f64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_daxpy(
    _q: *mut c_void,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dscal(
    _q: *mut c_void,
    _n: i64,
    _alpha: f64,
    _x: *mut f64,
    _incx: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dcopy(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dswap(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_ddot(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _result: *mut f64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dnrm2(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut f64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_dasum(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut f64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_diamax(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_diamin(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut i64,
) -> i32 {
    STATUS_UNAVAILABLE
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_drot(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _c: f64,
    _s: f64,
) -> i32 {
    STATUS_UNAVAILABLE
}

pub unsafe extern "C" fn bf_saxpy_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: f32,
    _x: *const f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_sscal_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: f32,
    _x: *mut f32,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_scopy_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_sswap_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_sdot_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _result: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_snrm2_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_sasum_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_siamax_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_siamin_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_srot_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _c: f32,
    _s: f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_srotm_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _param: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_drotm_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _param: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_daxpy_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: f64,
    _x: *const f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dscal_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: f64,
    _x: *mut f64,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dcopy_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dswap_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_ddot_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _result: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dnrm2_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_dasum_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

pub unsafe extern "C" fn bf_diamax_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_diamin_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_drot_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _c: f64,
    _s: f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_caxpy_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: *const f32,
    _x: *const f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_cscal_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: *const f32,
    _x: *mut f32,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ccopy_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_cswap_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f32,
    _incx: i64,
    _y: *mut f32,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_cdotu_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _result: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_cdotc_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _y: *const f32,
    _incy: i64,
    _result: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_cnrm2_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_casum_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut f32,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ciamax_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f32,
    _incx: i64,
    _result: *mut i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zaxpy_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: *const f64,
    _x: *const f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zscal_async(
    _q: *mut c_void,
    _n: i64,
    _alpha: *const f64,
    _x: *mut f64,
    _incx: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zcopy_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zswap_async(
    _q: *mut c_void,
    _n: i64,
    _x: *mut f64,
    _incx: i64,
    _y: *mut f64,
    _incy: i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zdotu_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _result: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zdotc_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _y: *const f64,
    _incy: i64,
    _result: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_znrm2_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_zasum_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut f64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn bf_ziamax_async(
    _q: *mut c_void,
    _n: i64,
    _x: *const f64,
    _incx: i64,
    _result: *mut i64,
    _deps: *const *mut c_void,
    _n_deps: i64,
) -> *mut c_void {
    std::ptr::null_mut()
}
