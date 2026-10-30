//! The safe API against a real SYCL CPU (OpenCL) device. Without oneAPI the device tests skip
//! loudly, unless `BONEFISH_REQUIRE_SYCL=1` (the oneAPI CI job), which makes a skip a failure.
//! `without_sycl_everything_fails_cleanly` covers the other build and always runs. Nothing here
//! has run on a GPU.

use std::fmt::Debug;

use bonefish::{
    Complex, DeviceKind, Dft, Error, Placement, Queue, SYCL_AVAILABLE, Scalar, Transpose,
    UsmBuffer, UsmKind, blas,
};

/// A CPU queue, or `None` after a loud skip. With `BONEFISH_REQUIRE_SYCL=1` a build without
/// SYCL is a failure, not a skip.
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

#[test]
fn without_sycl_everything_fails_cleanly() {
    if SYCL_AVAILABLE {
        // Positive control: with the shim, the same call succeeds.
        assert!(Queue::new(DeviceKind::Cpu).is_ok());
        return;
    }
    for kind in [DeviceKind::Default, DeviceKind::Cpu, DeviceKind::Gpu] {
        match Queue::new(kind) {
            Err(Error::Unavailable(m)) => assert!(m.contains("without oneAPI"), "{m}"),
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }
}

#[test]
fn device_name_is_reported() {
    let Some(q) = queue("device_name") else {
        return;
    };
    assert!(!q.device_name().unwrap().is_empty());
}

#[test]
fn usm_roundtrips_for_every_kind_and_gates_host_access() {
    let Some(q) = queue("usm_roundtrips") else {
        return;
    };
    let data = [1.5_f64, -2.0, 3.25];
    for kind in [UsmKind::Shared, UsmKind::Device, UsmKind::Host] {
        let mut b =
            UsmBuffer::from_slice(&q, kind, &data).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        assert_eq!(b.len(), 3);
        assert_eq!(b.to_vec().unwrap(), data, "{kind:?}");
        match kind {
            UsmKind::Device => {
                assert!(b.as_slice().is_none() && b.as_mut_slice().is_none());
            }
            _ => {
                assert_eq!(b.as_slice().unwrap(), data);
                b.as_mut_slice().unwrap()[1] = 9.0;
                assert_eq!(b.to_vec().unwrap()[1], 9.0);
            }
        }
        b.copy_from_slice(&[7.0, 8.0, 9.0]).unwrap();
        assert_eq!(b.to_vec().unwrap(), [7.0, 8.0, 9.0]);
        assert!(matches!(
            b.copy_from_slice(&[1.0]),
            Err(Error::InvalidArgument(_))
        ));
    }
    let z = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 5).unwrap();
    assert_eq!(z.to_vec().unwrap(), [0.0; 5]);
    let e = UsmBuffer::<f32>::from_slice(&q, UsmKind::Shared, &[]).unwrap();
    assert!(e.is_empty() && e.to_vec().unwrap().is_empty());
}

#[test]
fn a_buffer_keeps_its_queue_alive() {
    let Some(q) = queue("buffer_keeps_queue") else {
        return;
    };
    let b = UsmBuffer::from_slice(&q, UsmKind::Shared, &[1.0_f32, 2.0]).unwrap();
    drop(q);
    // The original handle is gone; the buffer must still be usable and freeable.
    assert_eq!(b.to_vec().unwrap(), [1.0, 2.0]);
}

fn gemm_case<T>(q: &Queue)
where
    T: Scalar + From<u8> + Debug + PartialEq,
{
    let v = |xs: &[u8]| xs.iter().map(|x| T::from(*x)).collect::<Vec<T>>();
    // Column-major A = [1 3 5; 2 4 6] (2 x 3), B = [1 0; 0 1; 1 1] (3 x 2).
    // A*B = [1+5, 3+5; 2+6, 4+6] = [6 8; 8 10] -> column-major 6 8 8 10.
    let a = UsmBuffer::from_slice(q, UsmKind::Shared, &v(&[1, 2, 3, 4, 5, 6])).unwrap();
    let b = UsmBuffer::from_slice(q, UsmKind::Shared, &v(&[1, 0, 1, 0, 1, 1])).unwrap();
    let mut c = UsmBuffer::from_slice(q, UsmKind::Shared, &v(&[0, 0, 0, 0])).unwrap();
    blas::gemm(
        q,
        Transpose::No,
        Transpose::No,
        2,
        2,
        3,
        T::from(1),
        &a,
        2,
        &b,
        3,
        T::from(0),
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), v(&[6, 8, 8, 10]));

    // Negative control: A^T * A differs from A*B above, and beta = 1 accumulates onto C.
    // A^T*A (3x3) is [5 11 17; 11 25 39; 17 39 61] -> not the 2x2 above, so use C = 3x3 of ones.
    let mut c3 = UsmBuffer::from_slice(q, UsmKind::Shared, &v(&[1; 9])).unwrap();
    blas::gemm(
        q,
        Transpose::Trans,
        Transpose::No,
        3,
        3,
        2,
        T::from(1),
        &a,
        2,
        &a,
        2,
        T::from(1),
        &mut c3,
        3,
    )
    .unwrap();
    assert_eq!(
        c3.to_vec().unwrap(),
        v(&[6, 12, 18, 12, 26, 40, 18, 40, 62])
    );
}

#[test]
fn gemm_single_and_double_match_hand_values() {
    let Some(q) = queue("gemm_hand_values") else {
        return;
    };
    gemm_case::<f32>(&q);
    gemm_case::<f64>(&q);
}

#[test]
fn gemm_rejects_bad_shapes_before_touching_the_device() {
    let Some(q) = queue("gemm_rejects") else {
        return;
    };
    let a = UsmBuffer::from_slice(&q, UsmKind::Shared, &[1.0_f32; 4]).unwrap();
    let b = UsmBuffer::from_slice(&q, UsmKind::Shared, &[1.0_f32; 4]).unwrap();
    let sentinel = [42.0_f32; 4];
    let mut c = UsmBuffer::from_slice(&q, UsmKind::Shared, &sentinel).unwrap();
    // Positive control: the 2x2x2 call with exact buffers succeeds and does write C.
    blas::gemm(
        &q,
        Transpose::No,
        Transpose::No,
        2,
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
    assert_eq!(c.to_vec().unwrap(), [2.0; 4]);

    let mut c = UsmBuffer::from_slice(&q, UsmKind::Shared, &sentinel).unwrap();
    // 3x3x3 needs 9-element matrices; the buffers hold 4.
    let e = blas::gemm(
        &q,
        Transpose::No,
        Transpose::No,
        3,
        3,
        3,
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
    // lda below the row count.
    let e = blas::gemm(
        &q,
        Transpose::No,
        Transpose::No,
        2,
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
    assert!(matches!(e, Error::InvalidArgument(m) if m.contains("leading dimension")));
    // Neither rejected call may have written C: the validation ran first.
    assert_eq!(c.to_vec().unwrap(), sentinel);
}

#[test]
fn a_buffer_from_another_queue_is_rejected() {
    let Some(q1) = queue("queue_mismatch") else {
        return;
    };
    let q2 = Queue::new(DeviceKind::Cpu).unwrap();
    assert!(!q1.same_as(&q2) && q1.same_as(&q1.clone()));
    let a = UsmBuffer::from_slice(&q1, UsmKind::Shared, &[1.0_f32; 4]).unwrap();
    let b = UsmBuffer::from_slice(&q2, UsmKind::Shared, &[1.0_f32; 4]).unwrap();
    let mut c = UsmBuffer::from_slice(&q1, UsmKind::Shared, &[0.0_f32; 4]).unwrap();
    let e = blas::gemm(
        &q1,
        Transpose::No,
        Transpose::No,
        2,
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
    .unwrap_err();
    assert_eq!(e, Error::QueueMismatch);
}

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

#[test]
fn dft_forward_backward_and_normalised_round_trip() {
    let Some(q) = queue("dft_round_trip") else {
        return;
    };
    // Delta -> all ones; backward of ones is n * delta (unnormalised).
    let plan = Dft::<f64>::c2c(&q, &[4], Placement::InPlace, 1.0, 1.0).unwrap();
    assert_eq!(plan.len(), 4);
    let mut x = UsmBuffer::from_slice(
        &q,
        UsmKind::Shared,
        &[c(1., 0.), c(0., 0.), c(0., 0.), c(0., 0.)],
    )
    .unwrap();
    plan.forward_in_place(&mut x).unwrap();
    assert_eq!(x.to_vec().unwrap(), vec![c(1., 0.); 4]);
    plan.backward_in_place(&mut x).unwrap();
    assert_eq!(
        x.to_vec().unwrap(),
        [c(4., 0.), c(0., 0.), c(0., 0.), c(0., 0.)]
    );

    // Negative control: with backward scale 1/n the same round trip returns the input.
    let norm = Dft::<f64>::c2c(&q, &[4], Placement::InPlace, 1.0, 0.25).unwrap();
    let input = [c(1., 2.), c(-3., 0.5), c(0., -1.), c(2.5, 2.5)];
    let mut y = UsmBuffer::from_slice(&q, UsmKind::Shared, &input).unwrap();
    norm.forward_in_place(&mut y).unwrap();
    norm.backward_in_place(&mut y).unwrap();
    for (got, want) in y.to_vec().unwrap().iter().zip(input) {
        assert!(
            (got.re - want.re).abs() < 1e-12 && (got.im - want.im).abs() < 1e-12,
            "{got:?} vs {want:?}"
        );
    }
}

#[test]
fn dft_out_of_place_leaves_input_and_enforces_placement() {
    let Some(q) = queue("dft_out_of_place") else {
        return;
    };
    let plan = Dft::<f32>::c2c(&q, &[2, 3], Placement::NotInPlace, 1.0, 1.0).unwrap();
    let mut data = vec![Complex::new(0.0_f32, 0.0); 6];
    data[0] = Complex::new(1.0, 0.0);
    let input = UsmBuffer::from_slice(&q, UsmKind::Shared, &data).unwrap();
    let mut out = UsmBuffer::<Complex<f32>>::zeroed(&q, UsmKind::Shared, 6).unwrap();
    plan.forward(&input, &mut out).unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![Complex::new(1.0, 0.0); 6]);
    assert_eq!(input.to_vec().unwrap(), data, "the input must be untouched");

    // The wrong entry point for the plan's placement is an error, not a wrong answer.
    let mut scratch = UsmBuffer::<Complex<f32>>::zeroed(&q, UsmKind::Shared, 6).unwrap();
    assert!(matches!(
        plan.forward_in_place(&mut scratch),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn dft_rejects_short_buffers_and_empty_shapes() {
    let Some(q) = queue("dft_rejects") else {
        return;
    };
    let plan = Dft::<f64>::c2c(&q, &[8], Placement::InPlace, 1.0, 1.0).unwrap();
    let mut short = UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, 4).unwrap();
    assert!(matches!(
        plan.forward_in_place(&mut short),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        Dft::<f64>::c2c(&q, &[], Placement::InPlace, 1.0, 1.0),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        Dft::<f64>::c2c(&q, &[4, 0], Placement::InPlace, 1.0, 1.0),
        Err(Error::InvalidArgument(_))
    ));
    // Positive control: an exact-size buffer is accepted.
    let mut ok = UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, 8).unwrap();
    plan.forward_in_place(&mut ok).unwrap();
}
