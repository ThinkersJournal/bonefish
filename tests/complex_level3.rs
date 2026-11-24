//! Complex level 3 through the safe API (R3): `trsm`, `hemm`, `herk`, `her2k` on `Complex<f64>`.
//! Hand-computed values (derivations in comments); the triangle that must not be touched holds a poison value.
//! Needs a device; without oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`.
//! CPU/OpenCL device only; GPU UNVERIFIED.

#[path = "../bonefish-sys/tests/common/mod.rs"]
mod common;

use bonefish::level3::{hemm, her2k, herk, trsm};
use bonefish::{
    Complex, DeviceKind, Diag, Queue, SYCL_AVAILABLE, Side, Transpose, Uplo, UsmBuffer, UsmKind,
};

type Z = Complex<f64>;

fn z(r: f64, i: f64) -> Z {
    Complex::new(r, i)
}

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

fn buf(q: &Queue, data: &[Z]) -> UsmBuffer<Z> {
    UsmBuffer::from_slice(q, UsmKind::Shared, data).unwrap()
}

fn same(got: &UsmBuffer<Z>, want: &[Z], what: &str) {
    let got = got.to_vec().unwrap();
    let got = got.as_slice();
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g.re - w.re).abs() < 1e-12 && (g.im - w.im).abs() < 1e-12,
            "{what}: got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn trsm_solves_a_complex_upper_system() {
    common::arm();
    let Some(q) = queue("trsm") else { return };
    let p = z(99.0, 99.0);
    // A = [[2, 1+i], [0, 1]] (lower poisoned); X = [1+i, 2]; B = A X = [2(1+i) + 2(1+i), 2] = [4+4i, 2].
    let a = buf(&q, &[z(2.0, 0.0), p, z(1.0, 1.0), z(1.0, 0.0)]);
    let mut b = buf(&q, &[z(4.0, 4.0), z(2.0, 0.0)]);
    trsm(
        &q,
        Side::Left,
        Uplo::Upper,
        Transpose::No,
        Diag::NonUnit,
        2,
        1,
        z(1.0, 0.0),
        &a,
        2,
        &mut b,
        2,
    )
    .unwrap();
    same(&b, &[z(1.0, 1.0), z(2.0, 0.0)], "trsm");
}

#[test]
fn hemm_reads_only_the_upper_triangle() {
    common::arm();
    let Some(q) = queue("hemm") else { return };
    let p = z(99.0, 99.0);
    // A = [[2, 1+i], [1-i, 3]] stored upper; B = [1, i]; A B = [2 + (1+i)i, (1-i) + 3i] = [1+i, 1+2i].
    let a = buf(&q, &[z(2.0, 0.0), p, z(1.0, 1.0), z(3.0, 0.0)]);
    let b = buf(&q, &[z(1.0, 0.0), z(0.0, 1.0)]);
    let mut c = buf(&q, &[z(0.0, 0.0), z(0.0, 0.0)]);
    hemm(
        &q,
        Side::Left,
        Uplo::Upper,
        2,
        1,
        z(1.0, 0.0),
        &a,
        2,
        &b,
        2,
        z(0.0, 0.0),
        &mut c,
        2,
    )
    .unwrap();
    same(&c, &[z(1.0, 1.0), z(1.0, 2.0)], "hemm");
}

#[test]
fn herk_updates_only_the_upper_triangle_with_real_diagonal() {
    common::arm();
    let Some(q) = queue("herk") else { return };
    let p = z(99.0, 99.0);
    // A = [1+i, 2] (2x1); A A^H = [[2, 2+2i], [2-2i, 4]]; only upper written, lower stays poison.
    let a = buf(&q, &[z(1.0, 1.0), z(2.0, 0.0)]);
    let mut c = buf(&q, &[z(0.0, 0.0), p, z(0.0, 0.0), z(0.0, 0.0)]);
    herk(
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
    same(&c, &[z(2.0, 0.0), p, z(2.0, 2.0), z(4.0, 0.0)], "herk");
}

#[test]
fn her2k_uses_the_conjugate_of_alpha() {
    common::arm();
    let Some(q) = queue("her2k") else { return };
    let p = z(99.0, 99.0);
    // A = [1+i, 2], B = [1, i], alpha = i, k = 1.
    // alpha A B^H = [[-1+i, 1+i], [., 2]], conj(alpha) B A^H = [[-1-i, -2i], [., 2]]; sum upper = [-2, 1-i, 4].
    let a = buf(&q, &[z(1.0, 1.0), z(2.0, 0.0)]);
    let b = buf(&q, &[z(1.0, 0.0), z(0.0, 1.0)]);
    let mut c = buf(&q, &[z(0.0, 0.0), p, z(0.0, 0.0), z(0.0, 0.0)]);
    her2k(
        &q,
        Uplo::Upper,
        Transpose::No,
        2,
        1,
        z(0.0, 1.0),
        &a,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    same(&c, &[z(-2.0, 0.0), p, z(1.0, -1.0), z(4.0, 0.0)], "her2k");
}
