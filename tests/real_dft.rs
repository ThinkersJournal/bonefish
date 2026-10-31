//! `RealDft`: real <-> half-spectrum complex (R3b). Needs a device; without oneAPI the tests skip
//! loudly unless `BONEFISH_REQUIRE_SYCL=1`. CPU/OpenCL device only; the layout is UNVERIFIED on a GPU.

use bonefish::{Complex, DeviceKind, Error, Queue, RealDft, SYCL_AVAILABLE, UsmBuffer, UsmKind};
use std::f64::consts::PI;

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

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0.37 * i as f64).sin() + 0.25 * ((i * i) % 7) as f64 - 0.1 * i as f64)
        .collect()
}

/// Naive half spectrum, row-major, last dimension halved.
fn naive(x: &[f64], dims: &[usize]) -> Vec<Complex<f64>> {
    let d = dims.len();
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .map(|(i, n)| if i == d - 1 { n / 2 + 1 } else { *n })
        .collect();
    (0..out_dims.iter().product::<usize>())
        .map(|oi| {
            let (mut rem, mut k) = (oi, vec![0usize; d]);
            for a in (0..d).rev() {
                k[a] = rem % out_dims[a];
                rem /= out_dims[a];
            }
            let (mut re, mut im) = (0.0, 0.0);
            for (xi, v) in x.iter().enumerate() {
                let (mut rem, mut phase) = (xi, 0.0);
                for a in (0..d).rev() {
                    phase += (k[a] * (rem % dims[a])) as f64 / dims[a] as f64;
                    rem /= dims[a];
                }
                re += v * (-2.0 * PI * phase).cos();
                im += v * (-2.0 * PI * phase).sin();
            }
            Complex::new(re, im)
        })
        .collect()
}

fn max_err(got: &[Complex<f64>], want: &[Complex<f64>]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g.re - w.re).abs().max((g.im - w.im).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn forward_matches_the_naive_half_spectrum_in_one_two_and_three_dimensions() {
    let Some(q) = queue("forward") else { return };
    for dims in [
        vec![7_usize],
        vec![12],
        vec![4, 6],
        vec![5, 3],
        vec![2, 3, 4],
    ] {
        let plan = RealDft::<f64>::new(&q, &dims, 1.0, 1.0).unwrap();
        let x = signal(plan.real_len());
        let input = UsmBuffer::from_slice(&q, UsmKind::Shared, &x).unwrap();
        let mut out =
            UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, plan.complex_len()).unwrap();
        plan.forward(&input, &mut out).unwrap();
        let e = max_err(&out.to_vec().unwrap(), &naive(&x, &dims));
        assert!(e < 1e-9, "{dims:?}: max error {e}");
        // The sizes the plan reports are the layout's.
        assert_eq!(plan.real_len(), dims.iter().product::<usize>());
        assert_eq!(
            plan.complex_len(),
            dims[..dims.len() - 1].iter().product::<usize>() * (dims[dims.len() - 1] / 2 + 1)
        );
    }
}

#[test]
fn a_round_trip_with_the_inverse_scale_returns_the_input() {
    let Some(q) = queue("round_trip") else { return };
    for dims in [vec![9_usize], vec![4, 6]] {
        let n: usize = dims.iter().product();
        let plan = RealDft::<f64>::new(&q, &dims, 1.0, 1.0 / n as f64).unwrap();
        let x = signal(n);
        let input = UsmBuffer::from_slice(&q, UsmKind::Shared, &x).unwrap();
        let mut spec =
            UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, plan.complex_len()).unwrap();
        let mut back = UsmBuffer::<f64>::zeroed(&q, UsmKind::Shared, n).unwrap();
        plan.forward(&input, &mut spec).unwrap();
        plan.backward(&spec, &mut back).unwrap();
        for (i, (g, w)) in back.to_vec().unwrap().iter().zip(&x).enumerate() {
            assert!((g - w).abs() < 1e-9, "{dims:?} element {i}: {g} vs {w}");
        }
    }
}

#[test]
fn single_precision_works_within_float_tolerance() {
    let Some(q) = queue("f32") else { return };
    let plan = RealDft::<f32>::new(&q, &[8], 1.0, 1.0).unwrap();
    let x = signal(8);
    let xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    let input = UsmBuffer::from_slice(&q, UsmKind::Shared, &xf).unwrap();
    let mut out =
        UsmBuffer::<Complex<f32>>::zeroed(&q, UsmKind::Shared, plan.complex_len()).unwrap();
    plan.forward(&input, &mut out).unwrap();
    let got: Vec<Complex<f64>> = out
        .to_vec()
        .unwrap()
        .iter()
        .map(|c| Complex::new(f64::from(c.re), f64::from(c.im)))
        .collect();
    assert!(max_err(&got, &naive(&x, &[8])) < 1e-4);
}

#[test]
fn refusals_come_before_the_device_is_touched() {
    let Some(q) = queue("refusals") else { return };
    // No dimensions, a zero length.
    assert!(matches!(
        RealDft::<f64>::new(&q, &[], 1.0, 1.0),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        RealDft::<f64>::new(&q, &[4, 0], 1.0, 1.0),
        Err(Error::InvalidArgument(_))
    ));
    // Control: a valid plan builds.
    let plan = RealDft::<f64>::new(&q, &[8], 1.0, 1.0).unwrap();
    let ok_in = UsmBuffer::<f64>::zeroed(&q, UsmKind::Shared, 8).unwrap();
    let short_in = UsmBuffer::<f64>::zeroed(&q, UsmKind::Shared, 7).unwrap();
    let mut ok_out = UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, 5).unwrap();
    let mut short_out = UsmBuffer::<Complex<f64>>::zeroed(&q, UsmKind::Shared, 4).unwrap();
    assert!(plan.forward(&ok_in, &mut ok_out).is_ok());
    let e = plan.forward(&short_in, &mut ok_out).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("input has 7")),
        "{e}"
    );
    let e = plan.forward(&ok_in, &mut short_out).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("output has 4")),
        "{e}"
    );
    // A buffer from another queue.
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let foreign = UsmBuffer::<f64>::zeroed(&other, UsmKind::Shared, 8).unwrap();
    assert!(matches!(
        plan.forward(&foreign, &mut ok_out),
        Err(Error::QueueMismatch)
    ));
}
