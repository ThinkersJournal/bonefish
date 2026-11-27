// SPDX-License-Identifier: MIT OR Apache-2.0
//! Level 3 BLAS beyond `gemm` through the safe API (R3): `symm`, `syrk`, `syr2k`, `trmm`, `trsm`.
//! Hand-computed values; a triangle that must not be touched holds a poison value.
//! Needs a device; without oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`.
//! CPU/OpenCL device only; GPU UNVERIFIED.

#[path = "../bonefish-sys/tests/common/mod.rs"]
mod common;

use bonefish::level3::{symm, syr2k, syrk, trmm, trsm};
use bonefish::{
    DeviceKind, Diag, Error, Queue, SYCL_AVAILABLE, Side, Transpose, Uplo, UsmBuffer, UsmKind,
};

const P: f64 = 99.0;

fn queue(test: &str) -> Option<Queue> {
    if !SYCL_AVAILABLE {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but bonefish was built without oneAPI"
        );
        eprintln!("SKIP {test}: built without oneAPI. Nothing ran on a device.");
        return None;
    }
    Some(Queue::new(DeviceKind::Cpu).expect("a CPU SYCL device"))
}

fn buf<T: bonefish::Element>(q: &Queue, data: &[T]) -> UsmBuffer<T> {
    UsmBuffer::from_slice(q, UsmKind::Shared, data).unwrap()
}

#[test]
fn symm_left_and_right() {
    common::arm();
    let Some(q) = queue("symm") else { return };
    // A = [[2,1],[1,3]], B = [[1,2],[3,4]] (column-major data below).
    let b = buf(&q, &[1.0, 3.0, 2.0, 4.0]);
    let a_up = buf(&q, &[2.0, P, 1.0, 3.0]);
    let mut c = buf(&q, &[0.0; 4]);
    symm(
        &q,
        Side::Left,
        Uplo::Upper,
        2,
        2,
        1.0,
        &a_up,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![5.0, 10.0, 8.0, 14.0]); // A B
    let a_lo = buf(&q, &[2.0, 1.0, P, 3.0]);
    let mut c = buf(&q, &[1.0; 4]);
    // C = 2 B A + 1 * C.
    symm(
        &q,
        Side::Right,
        Uplo::Lower,
        2,
        2,
        2.0,
        &a_lo,
        2,
        &b,
        2,
        1.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![9.0, 21.0, 15.0, 31.0]); // 2 [[4,7],[10,15]] + 1
}

#[test]
fn syrk_and_syr2k_touch_only_the_named_triangle() {
    common::arm();
    let Some(q) = queue("syrk") else { return };
    // A = [1; 2] (2 x 1): A A^T = [[1,2],[2,4]].
    let a = buf(&q, &[1.0, 2.0]);
    let mut c = buf(&q, &[P; 4]);
    syrk(
        &q,
        Uplo::Upper,
        Transpose::No,
        2,
        1,
        1.0,
        &a,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![1.0, P, 2.0, 4.0]);
    // Transposed: A is 1 x 2 stored with ld 1; A^T A = [[1,2],[2,4]]; lower only.
    let mut c = buf(&q, &[P; 4]);
    syrk(
        &q,
        Uplo::Lower,
        Transpose::Trans,
        2,
        1,
        1.0,
        &a,
        1,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![1.0, 2.0, P, 4.0]);
    // syr2k: A = [1;2], B = [3;4]: A B^T + B A^T = [[6,10],[10,16]], upper.
    let b = buf(&q, &[3.0, 4.0]);
    let mut c = buf(&q, &[P; 4]);
    syr2k(
        &q,
        Uplo::Upper,
        Transpose::No,
        2,
        1,
        1.0,
        &a,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![6.0, P, 10.0, 16.0]);
}

#[test]
fn trmm_then_trsm_round_trips_and_unit_diag_ignores_the_stored_one() {
    common::arm();
    let Some(q) = queue("trmm_trsm") else { return };
    // A = [[2,1],[P,3]] upper.
    let a = buf(&q, &[2.0, P, 1.0, 3.0]);
    let mut b = buf(&q, &[1.0, 3.0, 2.0, 4.0]);
    trmm(
        &q,
        Side::Left,
        Uplo::Upper,
        Transpose::No,
        Diag::NonUnit,
        2,
        2,
        1.0,
        &a,
        2,
        &mut b,
        2,
    )
    .unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![5.0, 9.0, 8.0, 12.0]);
    trsm(
        &q,
        Side::Left,
        Uplo::Upper,
        Transpose::No,
        Diag::NonUnit,
        2,
        2,
        1.0,
        &a,
        2,
        &mut b,
        2,
    )
    .unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    // Unit diagonal: A = [[1,1],[0,1]], B := A B = [[4,6],[3,4]].
    let mut b = buf(&q, &[1.0, 3.0, 2.0, 4.0]);
    trmm(
        &q,
        Side::Left,
        Uplo::Upper,
        Transpose::No,
        Diag::Unit,
        2,
        2,
        1.0,
        &a,
        2,
        &mut b,
        2,
    )
    .unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![4.0, 3.0, 6.0, 4.0]);
    // Right side: B := B A with A = [[2,1],[0,3]] -> [[2, 1+6],[6, 3+12]] = [[2,7],[6,15]].
    let mut b = buf(&q, &[1.0, 3.0, 2.0, 4.0]);
    trmm(
        &q,
        Side::Right,
        Uplo::Upper,
        Transpose::No,
        Diag::NonUnit,
        2,
        2,
        1.0,
        &a,
        2,
        &mut b,
        2,
    )
    .unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![2.0, 6.0, 7.0, 15.0]);
}

#[test]
fn single_precision_works() {
    common::arm();
    let Some(q) = queue("level3_f32") else { return };
    let p = P as f32;
    let a = buf(&q, &[2.0_f32, p, 1.0, 3.0]);
    let b = buf(&q, &[1.0_f32, 3.0, 2.0, 4.0]);
    let mut c = buf(&q, &[0.0_f32; 4]);
    symm(
        &q,
        Side::Left,
        Uplo::Upper,
        2,
        2,
        1.0,
        &a,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![5.0_f32, 10.0, 8.0, 14.0]);
}

#[test]
fn bad_extents_are_refused_in_rust_and_the_good_case_passes() {
    common::arm();
    let Some(q) = queue("level3_errors") else {
        return;
    };
    let a = buf(&q, &[2.0, P, 1.0, 3.0]);
    let b = buf(&q, &[1.0, 3.0, 2.0, 4.0]);
    let mut c = buf(&q, &[0.0; 4]);
    // Positive control.
    symm(
        &q,
        Side::Left,
        Uplo::Upper,
        2,
        2,
        1.0,
        &a,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    // C too short for 3 x 2.
    let e = symm(
        &q,
        Side::Left,
        Uplo::Upper,
        3,
        2,
        1.0,
        &a,
        3,
        &b,
        3,
        0.0,
        &mut c,
        3,
    )
    .unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // lda below the row count of A.
    let e = symm(
        &q,
        Side::Left,
        Uplo::Upper,
        2,
        2,
        1.0,
        &a,
        1,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // The right-side A is n x n, so m = 1, n = 3 needs a 3 x 3 A, which `a` is not.
    let e = trsm(
        &q,
        Side::Right,
        Uplo::Upper,
        Transpose::No,
        Diag::NonUnit,
        1,
        3,
        1.0,
        &a,
        3,
        &mut c,
        1,
    )
    .unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // syrk with k too large for A.
    let e = syrk(
        &q,
        Uplo::Upper,
        Transpose::No,
        2,
        3,
        1.0,
        &a,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // A buffer from another queue.
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let foreign = buf(&other, &[1.0; 4]);
    let e = symm(
        &q,
        Side::Left,
        Uplo::Upper,
        2,
        2,
        1.0,
        &a,
        2,
        &foreign,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap_err();
    assert!(matches!(e, Error::QueueMismatch), "{e}");
}
