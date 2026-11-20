//! Level 2 BLAS beyond `gemv` through the safe API (R3): `rotm`, `symv`, `syr`, `syr2`, `trmv`,
//! `trsv`, `gbmv`. Hand-computed values; a triangle that must not be touched holds a poison value.
//! Needs a device; without oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`.
//! CPU/OpenCL device only; GPU UNVERIFIED.

use bonefish::level2::{gbmv, rotm, symv, syr, syr2, trmv, trsv};
use bonefish::{
    DeviceKind, Diag, Error, Queue, SYCL_AVAILABLE, Transpose, Uplo, UsmBuffer, UsmKind,
};

const POISON: f64 = 99.0;

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

/// Symmetric `S = [[2,1,0],[1,3,4],[0,4,5]]`, one triangle stored, the other poisoned.
fn sym3(uplo: Uplo) -> Vec<f64> {
    let p = POISON;
    match uplo {
        Uplo::Upper => vec![2.0, p, p, 1.0, 3.0, p, 0.0, 4.0, 5.0],
        Uplo::Lower => vec![2.0, 1.0, 0.0, p, 3.0, 4.0, p, p, 5.0],
    }
}

#[test]
fn symv_reads_only_the_named_triangle() {
    let Some(q) = queue("symv") else { return };
    // S * [1,2,3] = [4,19,23]; y = 2 * that + 3 * [1,1,1].
    for uplo in [Uplo::Upper, Uplo::Lower] {
        let a = buf(&q, &sym3(uplo));
        let x = buf(&q, &[1.0, 2.0, 3.0]);
        let mut y = buf(&q, &[1.0; 3]);
        symv(&q, uplo, 3, 2.0, &a, 3, &x, 1, 3.0, &mut y, 1).unwrap();
        assert_eq!(y.to_vec().unwrap(), vec![11.0, 41.0, 49.0], "{uplo:?}");
    }
}

#[test]
fn symv_in_single_precision() {
    let Some(q) = queue("symv_f32") else { return };
    let p = POISON as f32;
    let a = buf(&q, &[2.0_f32, p, p, 1.0, 3.0, p, 0.0, 4.0, 5.0]);
    let x = buf(&q, &[1.0_f32, 2.0, 3.0]);
    let mut y = buf(&q, &[1.0_f32; 3]);
    symv(&q, Uplo::Upper, 3, 2.0, &a, 3, &x, 1, 3.0, &mut y, 1).unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![11.0_f32, 41.0, 49.0]);
}

#[test]
fn syr_updates_only_the_named_triangle() {
    let Some(q) = queue("syr") else { return };
    let x = buf(&q, &[1.0, 2.0, 3.0]);
    // A starts at 7 everywhere; += 2 x x^T on the upper triangle only.
    let mut a = buf(&q, &[7.0; 9]);
    syr(&q, Uplo::Upper, 3, 2.0, &x, 1, &mut a, 3).unwrap();
    assert_eq!(
        a.to_vec().unwrap(),
        vec![9.0, 7.0, 7.0, 11.0, 15.0, 7.0, 13.0, 19.0, 25.0]
    );
    // And the lower triangle for the other mode: the poison now sits in the upper triangle.
    let mut a = buf(&q, &[7.0; 9]);
    syr(&q, Uplo::Lower, 3, 2.0, &x, 1, &mut a, 3).unwrap();
    assert_eq!(
        a.to_vec().unwrap(),
        vec![9.0, 11.0, 13.0, 7.0, 15.0, 19.0, 7.0, 7.0, 25.0]
    );
}

#[test]
fn syr2_updates_only_the_named_triangle() {
    let Some(q) = queue("syr2") else { return };
    let (x, y) = (buf(&q, &[1.0, 2.0]), buf(&q, &[3.0, 4.0]));
    // x y^T + y x^T = [[6,10],[10,16]]; only the upper triangle is written.
    let mut a = buf(&q, &[0.0; 4]);
    syr2(&q, Uplo::Upper, 2, 1.0, &x, 1, &y, 1, &mut a, 2).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![6.0, 0.0, 10.0, 16.0]);
}

/// `T = [[1,2,3],[0,4,5],[0,0,6]]` stored upper, strictly lower poisoned.
fn tri3() -> Vec<f64> {
    let p = POISON;
    vec![1.0, p, p, 2.0, 4.0, p, 3.0, 5.0, 6.0]
}

#[test]
fn trmv_covers_notrans_trans_and_unit_diagonal() {
    let Some(q) = queue("trmv") else { return };
    let a = buf(&q, &tri3());
    for (trans, diag, want) in [
        (Transpose::No, Diag::NonUnit, [6.0, 9.0, 6.0]),
        (Transpose::Trans, Diag::NonUnit, [1.0, 6.0, 14.0]),
        // Unit: the stored diagonal (1, 4, 6) is ignored and taken as ones.
        (Transpose::No, Diag::Unit, [6.0, 6.0, 1.0]),
    ] {
        let mut x = buf(&q, &[1.0; 3]);
        trmv(&q, Uplo::Upper, trans, diag, 3, &a, 3, &mut x, 1).unwrap();
        assert_eq!(x.to_vec().unwrap(), want, "{trans:?} {diag:?}");
    }
}

#[test]
fn trsv_solves_exactly_and_undoes_trmv() {
    let Some(q) = queue("trsv") else { return };
    let a = buf(&q, &tri3());
    // z3 = 12/6 = 2, z2 = (13 - 5*2)/4 = 0.75, z1 = 14 - 2*0.75 - 3*2 = 6.5.
    let mut x = buf(&q, &[14.0, 13.0, 12.0]);
    trsv(
        &q,
        Uplo::Upper,
        Transpose::No,
        Diag::NonUnit,
        3,
        &a,
        3,
        &mut x,
        1,
    )
    .unwrap();
    assert_eq!(x.to_vec().unwrap(), vec![6.5, 0.75, 2.0]);
    // Through a scope: trmv then trsv straight after, with nothing waiting in between.
    let mut v = buf(&q, &[1.0, -2.0, 3.0]);
    q.scope(|s| {
        let (a, v) = (s.read(&a)?, s.write(&mut v)?);
        s.trmv(Uplo::Upper, Transpose::No, Diag::NonUnit, 3, &a, 3, &v, 1)?;
        s.trsv(Uplo::Upper, Transpose::No, Diag::NonUnit, 3, &a, 3, &v, 1)?;
        Ok(())
    })
    .unwrap();
    assert_eq!(v.to_vec().unwrap(), vec![1.0, -2.0, 3.0]);
}

#[test]
fn gbmv_tridiagonal_in_band_storage() {
    let Some(q) = queue("gbmv") else { return };
    // T = [[2,1,0],[3,4,1],[0,5,6]]: kl = ku = 1, lda = 3, entry (i,j) at ku + i - j + j * lda.
    let p = POISON;
    let a = buf(&q, &[p, 2.0, 3.0, 1.0, 4.0, 5.0, 1.0, 6.0, p]);
    let x = buf(&q, &[1.0, 2.0, 3.0]);
    // T x = [4,14,28]; y = 2 * that + 1.
    let mut y = buf(&q, &[1.0; 3]);
    gbmv(
        &q,
        Transpose::No,
        3,
        3,
        1,
        1,
        2.0,
        &a,
        3,
        &x,
        1,
        1.0,
        &mut y,
        1,
    )
    .unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![9.0, 29.0, 57.0]);
    // T^T x = [8,24,20].
    let mut y = buf(&q, &[0.0; 3]);
    gbmv(
        &q,
        Transpose::Trans,
        3,
        3,
        1,
        1,
        1.0,
        &a,
        3,
        &x,
        1,
        1.0,
        &mut y,
        1,
    )
    .unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![8.0, 24.0, 20.0]);
}

#[test]
fn rotm_applies_the_full_h_matrix() {
    let Some(q) = queue("rotm") else { return };
    // flag -1: full H. param = [flag, h11, h21, h12, h22] = [-1, 1, 3, 2, 4].
    let param = buf(&q, &[-1.0, 1.0, 3.0, 2.0, 4.0]);
    let (mut x, mut y) = (buf(&q, &[1.0, 2.0]), buf(&q, &[10.0, 20.0]));
    rotm(&q, 2, &mut x, 1, &mut y, 1, &param).unwrap();
    // x' = h11 x + h12 y, y' = h21 x + h22 y.
    assert_eq!(x.to_vec().unwrap(), vec![21.0, 42.0]);
    assert_eq!(y.to_vec().unwrap(), vec![43.0, 86.0]);
}

#[test]
fn bad_extents_are_refused_in_rust_and_the_good_case_passes() {
    let Some(q) = queue("level2_errors") else {
        return;
    };
    let a = buf(&q, &sym3(Uplo::Upper));
    let x = buf(&q, &[1.0, 2.0, 3.0]);
    let mut y = buf(&q, &[0.0; 3]);
    // Positive control: the same call with the right sizes succeeds.
    symv(&q, Uplo::Upper, 3, 1.0, &a, 3, &x, 1, 0.0, &mut y, 1).unwrap();
    // A matrix too short for n = 4.
    let e = symv(&q, Uplo::Upper, 4, 1.0, &a, 4, &x, 1, 0.0, &mut y, 1).unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // lda below n.
    let e = symv(&q, Uplo::Upper, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1).unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // A zero increment.
    let e = symv(&q, Uplo::Upper, 3, 1.0, &a, 3, &x, 0, 0.0, &mut y, 1).unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // rotm needs 5 parameters.
    let short = buf(&q, &[-1.0, 1.0, 3.0, 2.0]);
    let (mut p, mut r) = (buf(&q, &[1.0]), buf(&q, &[1.0]));
    let e = rotm(&q, 1, &mut p, 1, &mut r, 1, &short).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("5 elements")),
        "{e}"
    );
    // gbmv: lda below kl + ku + 1.
    let e = gbmv(
        &q,
        Transpose::No,
        3,
        3,
        1,
        1,
        1.0,
        &a,
        2,
        &x,
        1,
        0.0,
        &mut y,
        1,
    )
    .unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    // A buffer from another queue.
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let foreign = buf(&other, &[1.0; 3]);
    let e = symv(&q, Uplo::Upper, 3, 1.0, &a, 3, &foreign, 1, 0.0, &mut y, 1).unwrap_err();
    assert!(matches!(e, Error::QueueMismatch), "{e}");
}
