//! Level 1, `gemv` and DFT inside `Queue::scope` (R3c step 3). Needs a device; without oneAPI the
//! tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`. CPU/OpenCL device only; GPU UNVERIFIED.

use bonefish::{
    Complex, DeviceKind, Dft, Error, Placement, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer,
    UsmKind,
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

/// `y = 2x + y`, then `dot(x, y)` straight after, with nothing waiting in between. Measured: a
/// bare oneMKL `dot` submitted right after `axpy` read the old `y` in 299 of 300 rounds on an
/// in-order queue (`ROADMAP.md` R3c); the scope's chain is what makes this come out right, so this
/// is the end-to-end check of that, through the safe API.
#[test]
fn dot_straight_after_axpy_sees_the_axpy() {
    let Some(q) = queue("dot_after_axpy") else {
        return;
    };
    let x = buf(&q, &[1.0_f64, 2.0, 3.0, 4.0]);
    let mut y = buf(&q, &[1.0_f64; 4]);
    let mut r = buf(&q, &[0.0_f64]);
    for round in 0..50 {
        y.copy_from_slice(&[1.0; 4]).unwrap();
        q.scope(|s| {
            let x = s.read(&x)?;
            let y = s.write(&mut y)?;
            let r = s.write(&mut r)?;
            s.axpy(4, 2.0, &x, 1, &y, 1)?;
            s.dot(4, &x, 1, &y, 1, &r)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(r.to_vec().unwrap(), vec![70.0], "round {round}");
    }
}

#[test]
fn level_1_operations_give_the_right_numbers() {
    let Some(q) = queue("level_1") else { return };
    let x = buf(&q, &[1.0_f64, -5.0, 3.0, 4.0]);
    let mut y = buf(&q, &[1.0_f64, 1.0, 1.0, 1.0]);
    let mut z = buf(&q, &[0.0_f64; 4]);
    let mut dot = buf(&q, &[0.0_f64]);
    let mut nrm = buf(&q, &[0.0_f64]);
    let mut asum = buf(&q, &[0.0_f64]);
    let mut imax = buf(&q, &[-1_i64]);
    q.scope(|s| {
        let x = s.read(&x)?;
        let y = s.write(&mut y)?;
        let z = s.write(&mut z)?;
        let (dot, nrm, asum, imax) = (
            s.write(&mut dot)?,
            s.write(&mut nrm)?,
            s.write(&mut asum)?,
            s.write(&mut imax)?,
        );
        s.axpy(4, 2.0, &x, 1, &y, 1)?; // y = [3, -9, 7, 9]
        s.dot(4, &x, 1, &y, 1, &dot)?; // 3 + 45 + 21 + 36 = 105
        s.nrm2(4, &x, 1, &nrm)?; // sqrt(1 + 25 + 9 + 16) = sqrt(51)
        s.asum(4, &x, 1, &asum)?; // 13
        s.iamax(4, &x, 1, &imax)?; // index 1 (|-5|), zero based
        s.copy(4, &y, 1, &z, 1)?; // z = y
        s.scal(4, 10.0, &z, 1)?; // z = [30, -90, 70, 90]
        Ok(())
    })
    .unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![3.0, -9.0, 7.0, 9.0]);
    assert_eq!(dot.to_vec().unwrap(), vec![105.0]);
    assert!((nrm.to_vec().unwrap()[0] - 51.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(asum.to_vec().unwrap(), vec![13.0]);
    assert_eq!(imax.to_vec().unwrap(), vec![1]);
    assert_eq!(z.to_vec().unwrap(), vec![30.0, -90.0, 70.0, 90.0]);
}

#[test]
fn swap_and_strided_gemv_give_the_right_numbers() {
    let Some(q) = queue("swap_gemv") else { return };
    let mut a = buf(&q, &[1.0_f32, 2.0, 3.0]);
    let mut b = buf(&q, &[9.0_f32, 8.0, 7.0]);
    // A = [[1, 3], [2, 4]] column-major, x with stride 2 = [1, 1], y = A x.
    let m = buf(&q, &[1.0_f32, 2.0, 3.0, 4.0]);
    let x = buf(&q, &[1.0_f32, 0.0, 1.0]);
    let mut y = buf(&q, &[0.0_f32; 2]);
    let mut yt = buf(&q, &[0.0_f32; 2]);
    q.scope(|s| {
        let (a, b) = (s.write(&mut a)?, s.write(&mut b)?);
        s.swap(3, &a, 1, &b, 1)?;
        let (m, x) = (s.read(&m)?, s.read(&x)?);
        let (y, yt) = (s.write(&mut y)?, s.write(&mut yt)?);
        s.gemv(Transpose::No, 2, 2, 1.0, &m, 2, &x, 2, 0.0, &y, 1)?;
        s.gemv(Transpose::Trans, 2, 2, 1.0, &m, 2, &x, 2, 0.0, &yt, 1)?;
        Ok(())
    })
    .unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![9.0, 8.0, 7.0]);
    assert_eq!(b.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(y.to_vec().unwrap(), vec![4.0, 6.0]);
    assert_eq!(yt.to_vec().unwrap(), vec![3.0, 7.0]);
}

#[test]
fn level_1_refusals_happen_before_anything_is_submitted() {
    let Some(q) = queue("refusals") else { return };
    let x = buf(&q, &[1.0_f64; 4]);
    let mut y = buf(&q, &[1.0_f64; 4]);
    let mut r = buf(&q, &[0.0_f64; 1]);
    let mut short = buf(&q, &[0.0_f64; 3]);
    q.scope(|s| {
        let x = s.read(&x)?;
        let y = s.write(&mut y)?;
        let r = s.write(&mut r)?;
        let short = s.write(&mut short)?;
        // Control: a valid call is accepted.
        assert!(s.axpy(4, 1.0, &x, 1, &y, 1).is_ok());
        // A buffer that is too short, a zero increment, a stride that runs past the end.
        let e = s.axpy(4, 1.0, &x, 1, &short, 1).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("need 4")),
            "{e}"
        );
        let e = s.axpy(4, 1.0, &x, 0, &y, 1).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("at least 1")),
            "{e}"
        );
        let e = s.axpy(3, 1.0, &x, 2, &y, 1).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("stride 2")),
            "{e}"
        );
        // The output is also an input: y as its own x, and a dot result that is one of its inputs.
        let e = s.axpy(4, 1.0, &y, 1, &y, 1).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("same buffer")),
            "{e}"
        );
        let e = s.swap(4, &y, 1, &y, 1).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("same buffer")),
            "{e}"
        );
        let e = s.dot(1, &r, 1, &x, 1, &r).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("same buffer")),
            "{e}"
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn an_in_place_dft_transforms_a_delta() {
    let Some(q) = queue("dft_in_place") else {
        return;
    };
    let dft = Dft::<f64>::c2c(&q, &[4], Placement::InPlace, 1.0, 1.0).unwrap();
    // A 4-point delta.
    let mut data = buf(
        &q,
        &[
            Complex::new(1.0_f64, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
    );
    q.scope(|s| {
        let plan = s.plan(&dft)?;
        let d = s.write(&mut data)?;
        s.dft_forward_in_place(&plan, &d)?;
        Ok(())
    })
    .unwrap();
    for c in data.to_vec().unwrap() {
        assert!((c.re - 1.0).abs() < 1e-12 && c.im.abs() < 1e-12, "{c:?}");
    }
    // Forward then backward in one scope returns to n * input (unscaled transforms).
    q.scope(|s| {
        let plan = s.plan(&dft)?;
        let d = s.write(&mut data)?;
        s.dft_backward_in_place(&plan, &d)?;
        Ok(())
    })
    .unwrap();
    let v = data.to_vec().unwrap();
    assert!(
        (v[0].re - 4.0).abs() < 1e-12 && v[1].re.abs() < 1e-12,
        "{v:?}"
    );
}

#[test]
fn an_out_of_place_dft_leaves_its_input_alone_and_refuses_wrong_use() {
    let Some(q) = queue("dft_out_of_place") else {
        return;
    };
    let dft = Dft::<f32>::c2c(&q, &[4], Placement::NotInPlace, 1.0, 1.0).unwrap();
    let inplace = Dft::<f32>::c2c(&q, &[4], Placement::InPlace, 1.0, 1.0).unwrap();
    let delta = [
        Complex::new(1.0_f32, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    let input = buf(&q, &delta);
    let mut output = buf(&q, &[Complex::new(0.0_f32, 0.0); 4]);
    let mut small = buf(&q, &[Complex::new(0.0_f32, 0.0); 2]);
    q.scope(|s| {
        let plan = s.plan(&dft)?;
        let wrong = s.plan(&inplace)?;
        let i = s.read(&input)?;
        let o = s.write(&mut output)?;
        let sm = s.write(&mut small)?;
        // Refusals: wrong placement both ways, a short output, output == input.
        let e = s.dft_forward_in_place(&plan, &o).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("out-of-place")),
            "{e}"
        );
        let e = s.dft_forward(&wrong, &i, &o).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("in-place")),
            "{e}"
        );
        let e = s.dft_forward(&plan, &i, &sm).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("needs 4")),
            "{e}"
        );
        let e = s.dft_forward(&plan, &o, &o).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("same buffer")),
            "{e}"
        );
        // Control: the right call is accepted.
        s.dft_forward(&plan, &i, &o)?;
        Ok(())
    })
    .unwrap();
    assert_eq!(input.to_vec().unwrap(), delta.to_vec());
    for c in output.to_vec().unwrap() {
        assert!((c.re - 1.0).abs() < 1e-6 && c.im.abs() < 1e-6, "{c:?}");
    }
}

#[test]
fn a_plan_from_another_queue_is_rejected_at_binding() {
    let Some(q) = queue("plan_mismatch") else {
        return;
    };
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let foreign = Dft::<f32>::c2c(&other, &[4], Placement::InPlace, 1.0, 1.0).unwrap();
    let own = Dft::<f32>::c2c(&q, &[4], Placement::InPlace, 1.0, 1.0).unwrap();
    q.scope(|s| {
        assert!(s.plan(&own).is_ok());
        assert!(matches!(s.plan(&foreign), Err(Error::QueueMismatch)));
        Ok(())
    })
    .unwrap();
}
