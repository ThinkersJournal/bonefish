//! The `Level3` and `Hermitian3` implementations for `Complex<f32>` and `Complex<f64>` (moved out
//! of `level3.rs`, R15: they are two long macro bodies of the same shape as the real ones).

use std::ffi::c_void;

use bonefish_sys as sys;

use super::{Hermitian3, Level3};
use crate::scalar::Complex;

macro_rules! impl_level3_complex_c {
    () => {
        impl Level3 for Complex<f32> {
            unsafe fn symm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: `Complex<f32>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two
                // `f32`; `alpha` and `beta` outlive the call. The rest is the caller's contract.
                unsafe {
                    sys::bf_csymm_async(
                        q,
                        side,
                        uplo,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn syrk_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_csyrk_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn syr2k_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_csyr2k_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn trmm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                trans: i32,
                diag: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *mut Self,
                ldb: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_ctrmm_async(
                        q,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn trsm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                trans: i32,
                diag: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *mut Self,
                ldb: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_ctrsm_async(
                        q,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        deps,
                        n_deps,
                    )
                }
            }
        }

        impl Hermitian3 for Complex<f32> {
            type Real = f32;
            unsafe fn hemm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_chemm_async(
                        q,
                        side,
                        uplo,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn herk_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: f32,
                a: *const Self,
                lda: i64,
                beta: f32,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`; `alpha` and `beta` are real, by value.
                unsafe {
                    sys::bf_cherk_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        alpha,
                        a.cast(),
                        lda,
                        beta,
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn her2k_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: f32,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`; `beta` is real, by value.
                unsafe {
                    sys::bf_cher2k_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        beta,
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
        }
    };
}
impl_level3_complex_c!();

macro_rules! impl_level3_complex_z {
    () => {
        impl Level3 for Complex<f64> {
            unsafe fn symm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: `Complex<f64>` is `repr(C)` `(re, im)`, so a pointer to one is a pointer to two
                // `f64`; `alpha` and `beta` outlive the call. The rest is the caller's contract.
                unsafe {
                    sys::bf_zsymm_async(
                        q,
                        side,
                        uplo,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn syrk_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_zsyrk_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn syr2k_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_zsyr2k_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn trmm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                trans: i32,
                diag: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *mut Self,
                ldb: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_ztrmm_async(
                        q,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn trsm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                trans: i32,
                diag: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *mut Self,
                ldb: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_ztrsm_async(
                        q,
                        side,
                        uplo,
                        trans,
                        diag,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        deps,
                        n_deps,
                    )
                }
            }
        }

        impl Hermitian3 for Complex<f64> {
            type Real = f64;
            unsafe fn hemm_raw_async(
                q: *mut c_void,
                side: i32,
                uplo: i32,
                m: i64,
                n: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`.
                unsafe {
                    sys::bf_zhemm_async(
                        q,
                        side,
                        uplo,
                        m,
                        n,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        (&raw const beta).cast(),
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn herk_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: f64,
                a: *const Self,
                lda: i64,
                beta: f64,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`; `alpha` and `beta` are real, by value.
                unsafe {
                    sys::bf_zherk_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        alpha,
                        a.cast(),
                        lda,
                        beta,
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
            unsafe fn her2k_raw_async(
                q: *mut c_void,
                uplo: i32,
                trans: i32,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: f64,
                c: *mut Self,
                ldc: i64,
                deps: *const *mut c_void,
                n_deps: i64,
            ) -> *mut c_void {
                // SAFETY: as for `symm_raw_async`; `beta` is real, by value.
                unsafe {
                    sys::bf_zher2k_async(
                        q,
                        uplo,
                        trans,
                        n,
                        k,
                        (&raw const alpha).cast(),
                        a.cast(),
                        lda,
                        b.cast(),
                        ldb,
                        beta,
                        c.cast(),
                        ldc,
                        deps,
                        n_deps,
                    )
                }
            }
        }
    };
}
impl_level3_complex_z!();
