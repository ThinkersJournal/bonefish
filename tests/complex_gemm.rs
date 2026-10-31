//! Complex `gemm` (`Complex<f32>`, `Complex<f64>`) through `blas::gemm` and `Scope::gemm`. Needs a
//! device; without oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`. CPU/OpenCL device
//! only; nothing here has run on a GPU.

use bonefish::{
    Complex, DeviceKind, Error, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer, UsmKind, blas,
};

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

/// Deterministic complex values with both parts non-zero and no symmetry.
fn values(n: usize, seed: u32) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let t = (i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000;
            let u = (i as u32).wrapping_mul(40503).wrapping_add(seed * 7) % 1000;
            (f64::from(t) / 250.0 - 2.0, f64::from(u) / 250.0 - 2.0)
        })
        .collect()
}

fn mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Element (i, j) of `op(M)` where `M` is stored column-major with leading dimension `ld`.
fn at(m: &[(f64, f64)], ld: usize, op: Transpose, i: usize, j: usize) -> (f64, f64) {
    match op {
        Transpose::No => m[i + j * ld],
        Transpose::Trans => m[j + i * ld],
        Transpose::ConjTrans => {
            let v = m[j + i * ld];
            (v.0, -v.1)
        }
    }
}

/// Naive `alpha * op(A) * op(B) + beta * C`, column-major, `ldc = m`.
#[allow(clippy::too_many_arguments)]
fn naive(
    ta: Transpose,
    tb: Transpose,
    (m, n, k): (usize, usize, usize),
    alpha: (f64, f64),
    a: &[(f64, f64)],
    lda: usize,
    b: &[(f64, f64)],
    ldb: usize,
    beta: (f64, f64),
    c0: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let mut c = vec![(0.0, 0.0); m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = (0.0, 0.0);
            for p in 0..k {
                let t = mul(at(a, lda, ta, i, p), at(b, ldb, tb, p, j));
                acc = (acc.0 + t.0, acc.1 + t.1);
            }
            let lhs = mul(alpha, acc);
            let rhs = mul(beta, c0[i + j * m]);
            c[i + j * m] = (lhs.0 + rhs.0, lhs.1 + rhs.1);
        }
    }
    c
}

fn max_err(got: &[(f64, f64)], want: &[(f64, f64)]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g.0 - w.0).abs().max((g.1 - w.1).abs()))
        .fold(0.0, f64::max)
}

fn c64(v: &[(f64, f64)]) -> Vec<Complex<f64>> {
    v.iter().map(|(r, i)| Complex::new(*r, *i)).collect()
}
fn c32(v: &[(f64, f64)]) -> Vec<Complex<f32>> {
    v.iter()
        .map(|(r, i)| Complex::new(*r as f32, *i as f32))
        .collect()
}
fn back64(v: Vec<Complex<f64>>) -> Vec<(f64, f64)> {
    v.into_iter().map(|c| (c.re, c.im)).collect()
}
fn back32(v: Vec<Complex<f32>>) -> Vec<(f64, f64)> {
    v.into_iter()
        .map(|c| (f64::from(c.re), f64::from(c.im)))
        .collect()
}

const OPS: [Transpose; 3] = [Transpose::No, Transpose::Trans, Transpose::ConjTrans];

/// Stored shape of `op(M)` being `rows x cols`: the stored matrix is `rows x cols` for `No`,
/// `cols x rows` otherwise.
fn stored(op: Transpose, rows: usize, cols: usize) -> (usize, usize) {
    if op == Transpose::No {
        (rows, cols)
    } else {
        (cols, rows)
    }
}

#[test]
fn double_precision_matches_the_naive_product_for_every_op_pair() {
    let Some(q) = queue("zgemm") else { return };
    let (m, n, k) = (5, 4, 3);
    let (alpha, beta) = ((0.5, -2.0), (1.0, 1.0)); // both parts non-zero
    for ta in OPS {
        for tb in OPS {
            let (ar, ac) = stored(ta, m, k);
            let (br, bc) = stored(tb, k, n);
            let (a, b, c0) = (values(ar * ac, 1), values(br * bc, 2), values(m * n, 3));
            let want = naive(ta, tb, (m, n, k), alpha, &a, ar, &b, br, beta, &c0);
            let abuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&a)).unwrap();
            let bbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&b)).unwrap();
            let mut cbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&c0)).unwrap();
            blas::gemm(
                &q,
                ta,
                tb,
                m,
                n,
                k,
                Complex::new(alpha.0, alpha.1),
                &abuf,
                ar,
                &bbuf,
                br,
                Complex::new(beta.0, beta.1),
                &mut cbuf,
                m,
            )
            .unwrap();
            let e = max_err(&back64(cbuf.to_vec().unwrap()), &want);
            assert!(e < 1e-12, "{ta:?} x {tb:?}: max error {e}");
        }
    }
}

#[test]
fn single_precision_matches_within_float_tolerance() {
    let Some(q) = queue("cgemm") else { return };
    let (m, n, k) = (5, 4, 3);
    let (alpha, beta) = ((0.5, -2.0), (1.0, 1.0));
    for (ta, tb) in [
        (Transpose::No, Transpose::No),
        (Transpose::ConjTrans, Transpose::Trans),
    ] {
        let (ar, ac) = stored(ta, m, k);
        let (br, bc) = stored(tb, k, n);
        let (a, b, c0) = (values(ar * ac, 4), values(br * bc, 5), values(m * n, 6));
        let want = naive(ta, tb, (m, n, k), alpha, &a, ar, &b, br, beta, &c0);
        let abuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c32(&a)).unwrap();
        let bbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c32(&b)).unwrap();
        let mut cbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c32(&c0)).unwrap();
        blas::gemm(
            &q,
            ta,
            tb,
            m,
            n,
            k,
            Complex::new(alpha.0 as f32, alpha.1 as f32),
            &abuf,
            ar,
            &bbuf,
            br,
            Complex::new(beta.0 as f32, beta.1 as f32),
            &mut cbuf,
            m,
        )
        .unwrap();
        let e = max_err(&back32(cbuf.to_vec().unwrap()), &want);
        assert!(e < 1e-4, "{ta:?} x {tb:?}: max error {e}");
    }
}

/// Negative controls for the reference and the wiring: `ConjTrans` must differ from `Trans` on
/// data with imaginary parts (else a missing conjugation would pass), and a swapped (re, im) pair
/// in `alpha` must change the answer (else the pointer hand-off could be wrong unnoticed).
#[test]
fn conjugation_and_the_scalar_halves_are_not_interchangeable() {
    let Some(q) = queue("controls") else { return };
    let (m, n, k) = (3, 3, 3);
    let (a, b, c0) = (values(9, 7), values(9, 8), vec![(0.0, 0.0); 9]);
    let one = (1.0, 0.0);
    let run = |ta: Transpose, alpha: (f64, f64)| {
        let abuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&a)).unwrap();
        let bbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&b)).unwrap();
        let mut cbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&c0)).unwrap();
        blas::gemm(
            &q,
            ta,
            Transpose::No,
            m,
            n,
            k,
            Complex::new(alpha.0, alpha.1),
            &abuf,
            3,
            &bbuf,
            3,
            Complex::new(0.0, 0.0),
            &mut cbuf,
            3,
        )
        .unwrap();
        back64(cbuf.to_vec().unwrap())
    };
    let trans = run(Transpose::Trans, one);
    let conj = run(Transpose::ConjTrans, one);
    assert!(
        max_err(&trans, &conj) > 0.1,
        "Trans and ConjTrans gave the same answer on complex data"
    );
    // Each agrees with its own naive reference (so the difference is the right one).
    let want_conj = naive(
        Transpose::ConjTrans,
        Transpose::No,
        (m, n, k),
        one,
        &a,
        3,
        &b,
        3,
        (0.0, 0.0),
        &c0,
    );
    assert!(max_err(&conj, &want_conj) < 1e-12);
    // alpha = (2, 3) versus the swapped (3, 2).
    let x = run(Transpose::No, (2.0, 3.0));
    let y = run(Transpose::No, (3.0, 2.0));
    assert!(
        max_err(&x, &y) > 0.1,
        "swapping alpha's parts changed nothing"
    );
    let want_x = naive(
        Transpose::No,
        Transpose::No,
        (m, n, k),
        (2.0, 3.0),
        &a,
        3,
        &b,
        3,
        (0.0, 0.0),
        &c0,
    );
    assert!(max_err(&x, &want_x) < 1e-12);
}

/// The scoped, non-blocking form gives the same answer as the blocking one.
#[test]
fn the_scoped_form_matches_the_blocking_one() {
    let Some(q) = queue("scoped") else { return };
    let (m, n, k) = (4, 5, 3);
    let (alpha, beta) = ((1.5, 0.25), (0.0, 1.0));
    let (a, b, c0) = (values(m * k, 9), values(k * n, 10), values(m * n, 11));
    let want = naive(
        Transpose::No,
        Transpose::No,
        (m, n, k),
        alpha,
        &a,
        m,
        &b,
        k,
        beta,
        &c0,
    );
    let abuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&a)).unwrap();
    let bbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&b)).unwrap();
    let mut cbuf = UsmBuffer::from_slice(&q, UsmKind::Shared, &c64(&c0)).unwrap();
    q.scope(|s| {
        let (a, b) = (s.read(&abuf)?, s.read(&bbuf)?);
        let c = s.write(&mut cbuf)?;
        s.gemm(
            Transpose::No,
            Transpose::No,
            m,
            n,
            k,
            Complex::new(alpha.0, alpha.1),
            &a,
            m,
            &b,
            k,
            Complex::new(beta.0, beta.1),
            &c,
            m,
        )?;
        Ok(())
    })
    .unwrap();
    assert!(max_err(&back64(cbuf.to_vec().unwrap()), &want) < 1e-12);
}

#[test]
fn refusals_are_the_same_as_for_real_gemm() {
    let Some(q) = queue("refusals") else { return };
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let ok = UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, 4).unwrap();
    let foreign = UsmBuffer::<Complex<f64>>::zeroed(&other, UsmKind::Shared, 4).unwrap();
    let mut c = UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, 4).unwrap();
    let one = Complex::new(1.0, 0.0);
    let zero = Complex::new(0.0, 0.0);
    let go = |a: &UsmBuffer<Complex<f64>>, lda: usize, c: &mut UsmBuffer<Complex<f64>>| {
        blas::gemm(
            &q,
            Transpose::No,
            Transpose::No,
            2,
            2,
            2,
            one,
            a,
            lda,
            &ok,
            2,
            zero,
            c,
            2,
        )
    };
    assert!(go(&ok, 2, &mut c).is_ok(), "control");
    let e = go(&ok, 1, &mut c).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("leading dimension")),
        "{e}"
    );
    assert!(matches!(go(&foreign, 2, &mut c), Err(Error::QueueMismatch)));
}
