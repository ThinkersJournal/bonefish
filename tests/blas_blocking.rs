//! The blocking level 1 and `gemv` functions in `bonefish::blas` (R3 step). Needs a device; without
//! oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`. CPU/OpenCL device only; GPU
//! UNVERIFIED.

use bonefish::{
    DeviceKind, Error, Queue, SYCL_AVAILABLE, Scalar, Transpose, UsmBuffer, UsmKind, blas,
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

fn buf<T: bonefish::Element>(q: &Queue, data: &[T]) -> UsmBuffer<T> {
    UsmBuffer::from_slice(q, UsmKind::Shared, data).unwrap()
}

/// All of level 1 and gemv for one precision, with hand-computed answers.
fn everything<T>(q: &Queue, from: fn(f64) -> T, to: fn(T) -> f64)
where
    T: Scalar,
{
    let f = |v: &[f64]| v.iter().map(|e| from(*e)).collect::<Vec<T>>();
    let back = |b: &UsmBuffer<T>| {
        b.to_vec()
            .unwrap()
            .into_iter()
            .map(to)
            .collect::<Vec<f64>>()
    };

    let x = buf(q, &f(&[1.0, -5.0, 3.0, 4.0]));
    let mut y = buf(q, &f(&[1.0, 1.0, 1.0, 1.0]));

    // axpy: y = 2x + y = [3, -9, 7, 9]
    blas::axpy(q, 4, from(2.0), &x, 1, &mut y, 1).unwrap();
    assert_eq!(back(&y), vec![3.0, -9.0, 7.0, 9.0]);

    // dot(x, y) = 3 + 45 + 21 + 36 = 105. (Each call here waits before it returns, so this does
    // not exercise the ordering hazard; `scope_ops.rs` does that with calls that do not wait.)
    assert_eq!(to(blas::dot(q, 4, &x, 1, &y, 1).unwrap()), 105.0);
    // nrm2(x) = sqrt(51), asum(x) = 13, iamax(x) = 1 (|-5|), zero-based.
    assert!((to(blas::nrm2(q, 4, &x, 1).unwrap()) - 51.0_f64.sqrt()).abs() < 1e-5);
    assert_eq!(to(blas::asum(q, 4, &x, 1).unwrap()), 13.0);
    assert_eq!(blas::iamax(q, 4, &x, 1).unwrap(), 1);

    // copy, scal, swap
    let mut z = buf(q, &f(&[0.0; 4]));
    blas::copy(q, 4, &y, 1, &mut z, 1).unwrap();
    blas::scal(q, 4, from(10.0), &mut z, 1).unwrap();
    assert_eq!(back(&z), vec![30.0, -90.0, 70.0, 90.0]);
    blas::swap(q, 4, &mut y, 1, &mut z, 1).unwrap();
    assert_eq!(back(&y), vec![30.0, -90.0, 70.0, 90.0]);
    assert_eq!(back(&z), vec![3.0, -9.0, 7.0, 9.0]);

    // gemv with A = [[1, 3], [2, 4]] column-major and x with stride 2 = [1, 1].
    let a = buf(q, &f(&[1.0, 2.0, 3.0, 4.0]));
    let v = buf(q, &f(&[1.0, 0.0, 1.0]));
    let mut yn = buf(q, &f(&[0.0, 0.0]));
    let mut yt = buf(q, &f(&[0.0, 0.0]));
    blas::gemv(
        q,
        Transpose::No,
        2,
        2,
        from(1.0),
        &a,
        2,
        &v,
        2,
        from(0.0),
        &mut yn,
        1,
    )
    .unwrap();
    blas::gemv(
        q,
        Transpose::Trans,
        2,
        2,
        from(1.0),
        &a,
        2,
        &v,
        2,
        from(0.0),
        &mut yt,
        1,
    )
    .unwrap();
    assert_eq!(back(&yn), vec![4.0, 6.0]);
    assert_eq!(back(&yt), vec![3.0, 7.0]);
}

#[test]
fn level_1_and_gemv_double_precision() {
    let Some(q) = queue("blocking_f64") else {
        return;
    };
    everything::<f64>(&q, |v| v, |v| v);
}

#[test]
fn level_1_and_gemv_single_precision() {
    let Some(q) = queue("blocking_f32") else {
        return;
    };
    everything::<f32>(&q, |v| v as f32, f64::from);
}

#[test]
fn refusals_come_before_the_device_is_touched() {
    let Some(q) = queue("blocking_refusals") else {
        return;
    };
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let x = buf(&q, &[1.0_f64; 4]);
    let mut y = buf(&q, &[1.0_f64; 4]);
    let mut short = buf(&q, &[1.0_f64; 3]);
    let foreign = buf(&other, &[1.0_f64; 4]);

    // Control: a valid call works.
    assert!(blas::axpy(&q, 4, 1.0, &x, 1, &mut y, 1).is_ok());
    // A buffer too short, a zero increment, a buffer from another queue, an empty iamax.
    let e = blas::axpy(&q, 4, 1.0, &x, 1, &mut short, 1).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("need 4")),
        "{e}"
    );
    let e = blas::dot(&q, 4, &x, 0, &y, 1).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("at least 1")),
        "{e}"
    );
    assert!(matches!(
        blas::axpy(&q, 4, 1.0, &foreign, 1, &mut y, 1),
        Err(Error::QueueMismatch)
    ));
    assert!(matches!(
        blas::nrm2(&q, 4, &foreign, 1),
        Err(Error::QueueMismatch)
    ));
    let e = blas::iamax(&q, 0, &x, 1).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("zero elements")),
        "{e}"
    );
    // The buffers the refused calls named are untouched.
    assert_eq!(short.to_vec().unwrap(), vec![1.0; 3]);
    assert_eq!(y.to_vec().unwrap(), vec![2.0; 4]); // only the valid axpy ran
}
