//! Complex level 1 and `gemv` (`Complex<f32>`, `Complex<f64>`) through `bonefish::blas` and `Scope`.
//! Needs a device; without oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`. CPU/OpenCL
//! device only; nothing here has run on a GPU.

use bonefish::{
    BlasElement, Complex, DeviceKind, Error, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer, UsmKind,
    blas,
};

type C = (f64, f64);

/// Run every op once, serially, for one element type.
///
/// WORKAROUND, not a fix (ROADMAP R12). Without this, running these tests on parallel threads
/// crashed 11 runs in 100 on Windows (oneAPI 2026.1, CPU device; 0xC06D007F "procedure not found"
/// raised by `mkl_sycl_blas.6.dll`'s delay-load of `mkl_tbb_thread.3.dll!mkl_serv_verbose_mode`,
/// then an access violation executing an address in no module). With every op run once serially
/// first it was 0 in 100. `RUST_TEST_THREADS=1` also gave 0 in 100. The shim serialising every
/// call did NOT remove the crash, so the mechanism is not shown; only the cold-start dependence is.
fn warm<T: BlasElement + Copy>(q: &Queue, one: T) {
    let v = UsmBuffer::from_slice(q, UsmKind::Shared, &[one; 4]).unwrap();
    let mut y = UsmBuffer::from_slice(q, UsmKind::Shared, &[one; 4]).unwrap();
    let mut z = UsmBuffer::from_slice(q, UsmKind::Shared, &[one; 4]).unwrap();
    blas::axpy(q, 4, one, &v, 1, &mut y, 1).unwrap();
    blas::scal(q, 4, one, &mut y, 1).unwrap();
    blas::copy(q, 4, &v, 1, &mut y, 1).unwrap();
    blas::swap(q, 4, &mut z, 1, &mut y, 1).unwrap();
    blas::dot(q, 4, &v, 1, &y, 1).unwrap();
    blas::dotc(q, 4, &v, 1, &y, 1).unwrap();
    blas::nrm2(q, 4, &v, 1).unwrap();
    blas::asum(q, 4, &v, 1).unwrap();
    blas::iamax(q, 4, &v, 1).unwrap();
    blas::gemv(q, Transpose::No, 2, 2, one, &v, 2, &z, 1, one, &mut y, 1).unwrap();
}
fn warm_all(q: &Queue) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        warm(q, Complex::new(1.0f64, 0.0));
        warm(q, Complex::new(1.0f32, 0.0));
    });
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
    let q = Queue::new(DeviceKind::Cpu).expect("a CPU SYCL device");
    warm_all(&q);
    Some(q)
}

/// Deterministic complex values with both parts non-zero and no symmetry.
fn values(n: usize, seed: u32) -> Vec<C> {
    (0..n)
        .map(|i| {
            let t = (i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000;
            let u = (i as u32).wrapping_mul(40503).wrapping_add(seed * 7) % 1000;
            (f64::from(t) / 250.0 - 2.0, f64::from(u) / 250.0 - 2.0)
        })
        .collect()
}

fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn add(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}
fn conj(a: C) -> C {
    (a.0, -a.1)
}
fn c64(v: &[C]) -> Vec<Complex<f64>> {
    v.iter().map(|(r, i)| Complex::new(*r, *i)).collect()
}
fn c32(v: &[C]) -> Vec<Complex<f32>> {
    v.iter()
        .map(|(r, i)| Complex::new(*r as f32, *i as f32))
        .collect()
}
fn back64(v: Vec<Complex<f64>>) -> Vec<C> {
    v.into_iter().map(|c| (c.re, c.im)).collect()
}
fn back32(v: Vec<Complex<f32>>) -> Vec<C> {
    v.into_iter()
        .map(|c| (f64::from(c.re), f64::from(c.im)))
        .collect()
}
fn max_err(got: &[C], want: &[C]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g.0 - w.0).abs().max((g.1 - w.1).abs()))
        .fold(0.0, f64::max)
}
fn buf(q: &Queue, v: &[C]) -> UsmBuffer<Complex<f64>> {
    UsmBuffer::from_slice(q, UsmKind::Shared, &c64(v)).unwrap()
}
fn out(b: &UsmBuffer<Complex<f64>>) -> Vec<C> {
    back64(b.to_vec().unwrap())
}

/// Strided read: `n` elements of `v` starting at 0 with stride `inc`.
fn strided(v: &[C], n: usize, inc: usize) -> Vec<C> {
    (0..n).map(|i| v[i * inc]).collect()
}

#[test]
fn axpy_scal_copy_swap_match_the_reference_with_strides() {
    let Some(q) = queue("l1_basic") else { return };
    let n = 5;
    let (incx, incy) = (2, 1);
    let alpha = (0.5, -2.0);
    let (x, y) = (values(n * incx, 1), values(n * incy, 2));

    // axpy: y[i] += alpha * x[i * incx]
    let (xb, mut yb) = (buf(&q, &x), buf(&q, &y));
    blas::axpy(
        &q,
        n,
        Complex::new(alpha.0, alpha.1),
        &xb,
        incx,
        &mut yb,
        incy,
    )
    .unwrap();
    let want: Vec<C> = (0..n * incy)
        .map(|i| add(y[i], mul(alpha, strided(&x, n, incx)[i])))
        .collect();
    assert!(max_err(&out(&yb), &want) < 1e-12, "axpy");

    // scal: x[i * incx] *= alpha, and the elements in between are untouched
    let mut xb = buf(&q, &x);
    blas::scal(&q, n, Complex::new(alpha.0, alpha.1), &mut xb, incx).unwrap();
    let mut want = x.clone();
    for i in 0..n {
        want[i * incx] = mul(alpha, x[i * incx]);
    }
    assert!(
        max_err(&out(&xb), &want) < 1e-12,
        "scal (and the skipped elements stay put)"
    );

    // copy: y[i] = x[i * incx]
    let (xb, mut yb) = (buf(&q, &x), buf(&q, &vec![(0.0, 0.0); n]));
    blas::copy(&q, n, &xb, incx, &mut yb, 1).unwrap();
    assert!(max_err(&out(&yb), &strided(&x, n, incx)) < 1e-15, "copy");

    // swap: x[i * incx] <-> y[i]
    let (mut xb, mut yb) = (buf(&q, &x), buf(&q, &y[..n]));
    blas::swap(&q, n, &mut xb, incx, &mut yb, 1).unwrap();
    assert!(
        max_err(&out(&yb), &strided(&x, n, incx)) < 1e-15,
        "swap: y gets x"
    );
    let after = out(&xb);
    assert!(
        max_err(&strided(&after, n, incx), &y[..n]) < 1e-15,
        "swap: x gets y"
    );
}

/// `dot` is the plain product (BLAS `dotu`), `dotc` conjugates the first vector. They must DIFFER
/// on data with imaginary parts, or a missing conjugation would go unnoticed.
#[test]
fn dot_is_unconjugated_and_dotc_conjugates_the_first_vector() {
    let Some(q) = queue("dots") else { return };
    let n = 6;
    let (x, y) = (values(n, 3), values(n, 4));
    let (xb, yb) = (buf(&q, &x), buf(&q, &y));
    let want_u = x
        .iter()
        .zip(&y)
        .fold((0.0, 0.0), |a, (p, r)| add(a, mul(*p, *r)));
    let want_c = x
        .iter()
        .zip(&y)
        .fold((0.0, 0.0), |a, (p, r)| add(a, mul(conj(*p), *r)));
    let u = blas::dot(&q, n, &xb, 1, &yb, 1).unwrap();
    let c = blas::dotc(&q, n, &xb, 1, &yb, 1).unwrap();
    assert!(
        max_err(&[(u.re, u.im)], &[want_u]) < 1e-12,
        "dotu: {u:?} vs {want_u:?}"
    );
    assert!(
        max_err(&[(c.re, c.im)], &[want_c]) < 1e-12,
        "dotc: {c:?} vs {want_c:?}"
    );
    // The control: the two are genuinely different numbers here.
    assert!(
        max_err(&[(u.re, u.im)], &[(c.re, c.im)]) > 0.1,
        "dotu == dotc on complex data"
    );
}

/// `nrm2` and `asum` of COMPLEX data are REAL numbers (the annotation below fails to compile if the
/// result type is wrong). `asum` is `sum(|re| + |im|)`, not `sum |z|`, and the two differ here.
#[test]
fn nrm2_and_asum_are_real_and_asum_is_the_sum_of_parts() {
    let Some(q) = queue("norms") else { return };
    let n = 7;
    let x = values(n, 5);
    let xb = buf(&q, &x);
    let nrm: f64 = blas::nrm2(&q, n, &xb, 1).unwrap();
    let asum: f64 = blas::asum(&q, n, &xb, 1).unwrap();
    let want_nrm = x.iter().map(|z| z.0 * z.0 + z.1 * z.1).sum::<f64>().sqrt();
    let want_asum: f64 = x.iter().map(|z| z.0.abs() + z.1.abs()).sum();
    let sum_modulus: f64 = x.iter().map(|z| z.0.hypot(z.1)).sum();
    assert!((nrm - want_nrm).abs() < 1e-12, "nrm2 {nrm} vs {want_nrm}");
    assert!(
        (asum - want_asum).abs() < 1e-12,
        "asum {asum} vs {want_asum}"
    );
    assert!(
        (want_asum - sum_modulus).abs() > 0.1,
        "control: the data does not tell sum(|re|+|im|) from sum|z|"
    );
}

/// `iamax`: BLAS picks the element of largest `|re| + |im|`, not the largest modulus. Here element
/// 0 has the larger modulus (3.1 against 2.83) and element 1 the larger `|re|+|im|` (4 against
/// 3.1); the answer says which rule oneMKL follows. Zero-based, first index on ties.
#[test]
fn iamax_follows_the_sum_of_parts_rule() {
    let Some(q) = queue("iamax") else { return };
    let x = [(3.1, 0.0), (2.0, 2.0), (0.5, 0.5)];
    let xb = buf(&q, &x);
    let by_sum_of_parts = x
        .iter()
        .enumerate()
        .max_by(|a, b| (a.1.0.abs() + a.1.1.abs()).total_cmp(&(b.1.0.abs() + b.1.1.abs())))
        .unwrap()
        .0;
    let by_modulus = x
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.0.hypot(a.1.1).total_cmp(&b.1.0.hypot(b.1.1)))
        .unwrap()
        .0;
    assert_ne!(
        by_sum_of_parts, by_modulus,
        "control: the data must tell the two rules apart"
    );
    assert_eq!(
        blas::iamax(&q, 3, &xb, 1).unwrap(),
        by_sum_of_parts,
        "oneMKL did not follow the |re|+|im| rule (modulus rule would give {by_modulus})"
    );
    // Ties go to the first index.
    let tie = buf(&q, &[(1.0, 1.0), (2.0, 0.0), (0.0, 2.0)]);
    assert_eq!(blas::iamax(&q, 3, &tie, 1).unwrap(), 0);
}

#[test]
fn gemv_matches_the_reference_for_every_op() {
    let Some(q) = queue("gemv") else { return };
    let (m, n) = (5, 4);
    let (alpha, beta) = ((0.5, -2.0), (1.0, 1.0));
    let a = values(m * n, 6); // m x n, column-major, lda = m
    for op in [Transpose::No, Transpose::Trans, Transpose::ConjTrans] {
        let (xl, yl) = if op == Transpose::No { (n, m) } else { (m, n) };
        let (x, y0) = (values(xl, 7), values(yl, 8));
        let elem = |i: usize, j: usize| -> C {
            // element (i, j) of op(A), which is yl x xl
            match op {
                Transpose::No => a[i + j * m],
                Transpose::Trans => a[j + i * m],
                Transpose::ConjTrans => conj(a[j + i * m]),
            }
        };
        let want: Vec<C> = (0..yl)
            .map(|i| {
                let acc = (0..xl).fold((0.0, 0.0), |s, j| add(s, mul(elem(i, j), x[j])));
                add(mul(alpha, acc), mul(beta, y0[i]))
            })
            .collect();
        let (ab, xb, mut yb) = (buf(&q, &a), buf(&q, &x), buf(&q, &y0));
        blas::gemv(
            &q,
            op,
            m,
            n,
            Complex::new(alpha.0, alpha.1),
            &ab,
            m,
            &xb,
            1,
            Complex::new(beta.0, beta.1),
            &mut yb,
            1,
        )
        .unwrap();
        let e = max_err(&out(&yb), &want);
        assert!(e < 1e-12, "{op:?}: max error {e}");
    }
}

#[test]
fn single_precision_works_within_float_tolerance() {
    let Some(q) = queue("f32") else { return };
    let n = 6;
    let (x, y) = (values(n, 9), values(n, 10));
    let xb = UsmBuffer::from_slice(&q, UsmKind::Shared, &c32(&x)).unwrap();
    let yb = UsmBuffer::from_slice(&q, UsmKind::Shared, &c32(&y)).unwrap();
    let want = x
        .iter()
        .zip(&y)
        .fold((0.0, 0.0), |a, (p, r)| add(a, mul(conj(*p), *r)));
    let c = blas::dotc(&q, n, &xb, 1, &yb, 1).unwrap();
    assert!(max_err(&[(f64::from(c.re), f64::from(c.im))], &[want]) < 1e-4);
    let nrm: f32 = blas::nrm2(&q, n, &xb, 1).unwrap();
    let want_nrm = x.iter().map(|z| z.0 * z.0 + z.1 * z.1).sum::<f64>().sqrt();
    assert!((f64::from(nrm) - want_nrm).abs() < 1e-4);
    let mut zb = UsmBuffer::from_slice(&q, UsmKind::Shared, &c32(&y)).unwrap();
    blas::scal(&q, n, Complex::new(0.5f32, -2.0), &mut zb, 1).unwrap();
    let want_scal: Vec<C> = y.iter().map(|v| mul((0.5, -2.0), *v)).collect();
    assert!(max_err(&back32(zb.to_vec().unwrap()), &want_scal) < 1e-4);
}

/// The scoped, non-blocking forms, chained with nothing waiting in between: `axpy`, then `dot`
/// straight after it (the oneMKL call that ignores an in-order queue, ROADMAP R3c), then `nrm2`.
#[test]
fn scoped_complex_calls_chain_correctly() {
    let Some(q) = queue("scoped") else { return };
    let n = 6;
    let alpha = (2.0, 1.0);
    let (x, y) = (values(n, 11), values(n, 12));
    let xb = buf(&q, &x);
    let mut yb = buf(&q, &y);
    let mut dotu = buf(&q, &[(0.0, 0.0)]);
    let mut dotc = buf(&q, &[(0.0, 0.0)]);
    let mut nrm = UsmBuffer::<f64>::zeroed(&q, UsmKind::Shared, 1).unwrap();
    q.scope(|s| {
        let x = s.read(&xb)?;
        let (y, du, dc, r) = (
            s.write(&mut yb)?,
            s.write(&mut dotu)?,
            s.write(&mut dotc)?,
            s.write(&mut nrm)?,
        );
        s.axpy(n, Complex::new(alpha.0, alpha.1), &x, 1, &y, 1)?;
        s.dot(n, &x, 1, &y, 1, &du)?;
        s.dotc(n, &x, 1, &y, 1, &dc)?;
        s.nrm2(n, &y, 1, &r)?;
        Ok(())
    })
    .unwrap();
    let y1: Vec<C> = y
        .iter()
        .zip(&x)
        .map(|(yy, xx)| add(*yy, mul(alpha, *xx)))
        .collect();
    assert!(max_err(&out(&yb), &y1) < 1e-12);
    let wu = x
        .iter()
        .zip(&y1)
        .fold((0.0, 0.0), |a, (p, r)| add(a, mul(*p, *r)));
    let wc = x
        .iter()
        .zip(&y1)
        .fold((0.0, 0.0), |a, (p, r)| add(a, mul(conj(*p), *r)));
    assert!(
        max_err(&out(&dotu), &[wu]) < 1e-11,
        "dot straight after axpy"
    );
    assert!(
        max_err(&out(&dotc), &[wc]) < 1e-11,
        "dotc straight after axpy"
    );
    let wn = y1.iter().map(|z| z.0 * z.0 + z.1 * z.1).sum::<f64>().sqrt();
    assert!(
        (nrm.to_vec().unwrap()[0] - wn).abs() < 1e-11,
        "nrm2 straight after axpy"
    );
}

#[test]
fn refusals_are_the_same_as_for_real_data() {
    let Some(q) = queue("refusals") else { return };
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let x = buf(&q, &values(4, 13));
    let mut short = buf(&q, &values(3, 14));
    let foreign = UsmBuffer::from_slice(&other, UsmKind::Shared, &c64(&values(4, 15))).unwrap();
    let one = Complex::new(1.0, 0.0);
    let mut y = buf(&q, &values(4, 16));
    assert!(blas::axpy(&q, 4, one, &x, 1, &mut y, 1).is_ok(), "control");
    let e = blas::axpy(&q, 4, one, &x, 1, &mut short, 1).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("need 4")),
        "{e}"
    );
    assert!(matches!(
        blas::dotc(&q, 4, &foreign, 1, &x, 1),
        Err(Error::QueueMismatch)
    ));
    let e = blas::dot(&q, 4, &x, 0, &x, 1).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("at least 1")),
        "{e}"
    );
}
