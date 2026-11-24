//! Half-precision `gemm` through the safe API (`Scope::{hgemm, hgemm_f32, bgemm, bgemm_f32}`).
//! Needs a device; without oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`. A device
//! that refuses a variant is reported as NOT RUN, never silently skipped, and fails the test when
//! `BONEFISH_REQUIRE_HALF=1`. CPU/OpenCL device only; GPU UNVERIFIED.

#[path = "../bonefish-sys/tests/common/mod.rs"]
mod common;

use bonefish::half::{bf16, f16};
use bonefish::{DeviceKind, Error, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer, UsmKind};

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

// Column-major 2x2: A = [[1,2],[3,4]], B = [[1,0],[1,1]]; 2 * A * B = [[6,4],[14,8]].
const A: [f32; 4] = [1.0, 3.0, 2.0, 4.0];
const B: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
const WANT: [f32; 4] = [6.0, 14.0, 4.0, 8.0];

/// `Ok(true)` if the variant ran, `Ok(false)` (loudly) if the device refused it.
fn ran(name: &str, r: Result<(), Error>) -> bool {
    match r {
        Ok(()) => {
            eprintln!("RAN {name}");
            true
        }
        Err(e @ (Error::Sycl(_) | Error::Mkl(_) | Error::Other(_))) => {
            eprintln!("NOT RUN {name}: this device did not run the variant: {e}");
            assert!(
                !std::env::var("BONEFISH_REQUIRE_HALF").is_ok_and(|v| v == "1"),
                "{name}: BONEFISH_REQUIRE_HALF=1 but the variant failed: {e}"
            );
            false
        }
        Err(e) => panic!("{name}: {e}"),
    }
}

macro_rules! variant {
    ($test:ident, $method:ident, $a:ty, $c:ty, $conv:expr, $back:expr) => {
        #[test]
        fn $test() {
    common::arm();
            let Some(q) = queue(stringify!($test)) else {
                return;
            };
            let to: fn(f32) -> $a = $conv;
            let from: fn($c) -> f32 = $back;
            let a = buf(&q, &A.map(to));
            let b = buf(&q, &B.map(to));
            let mut c = buf(&q, &[<$c as bonefish::Element>::ZERO; 4]);
            let no = Transpose::No;
            let r = q.scope(|s| {
                let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
                s.$method(no, no, 2, 2, 2, 2.0, &a, 2, &b, 2, 0.0, &cw, 2)?;
                Ok(())
            });
            if ran(stringify!($method), r) {
                let got: Vec<f32> = c.to_vec().unwrap().into_iter().map(from).collect();
                assert_eq!(got, WANT.to_vec());
            }
        }
    };
}

variant!(
    hgemm_half_output,
    hgemm,
    f16,
    f16,
    f16::from_f32,
    f16::to_f32
);
variant!(hgemm_f32_output, hgemm_f32, f16, f32, f16::from_f32, |x| x);
variant!(
    bgemm_bf16_output,
    bgemm,
    bf16,
    bf16,
    bf16::from_f32,
    bf16::to_f32
);
variant!(
    bgemm_f32_output,
    bgemm_f32,
    bf16,
    f32,
    bf16::from_f32,
    |x| x
);

#[test]
fn extents_and_aliasing_are_checked() {
    common::arm();
    let Some(q) = queue("checks") else { return };
    let a = buf(&q, &A.map(f16::from_f32));
    let b = buf(&q, &B.map(f16::from_f32));
    let mut c = buf(&q, &[f16::ZERO; 4]);
    let no = Transpose::No;
    // lda 1 is below the 2 rows of A.
    let r = q.scope(|s| {
        let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
        s.hgemm(no, no, 2, 2, 2, 1.0, &a, 1, &b, 2, 0.0, &cw, 2)?;
        Ok(())
    });
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
    // A 3x3 product does not fit 4-element buffers.
    let r = q.scope(|s| {
        let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
        s.hgemm(no, no, 3, 3, 3, 1.0, &a, 3, &b, 3, 0.0, &cw, 3)?;
        Ok(())
    });
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
    // Same-buffer output is refused.
    let r = q.scope(|s| {
        let b2 = s.read(&b)?;
        let a2 = s.write(&mut c)?;
        s.hgemm(no, no, 2, 2, 2, 1.0, &a2, 2, &b2, 2, 0.0, &a2, 2)?;
        Ok(())
    });
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
    // Nothing ran: C is untouched.
    assert_eq!(c.to_vec().unwrap(), vec![f16::ZERO; 4]);
}

macro_rules! blocking {
    ($test:ident, $func:ident, $a:ty, $c:ty, $conv:expr, $back:expr) => {
        #[test]
        fn $test() {
    common::arm();
            let Some(q) = queue(stringify!($test)) else {
                return;
            };
            let to: fn(f32) -> $a = $conv;
            let from: fn($c) -> f32 = $back;
            let a = buf(&q, &A.map(to));
            let b = buf(&q, &B.map(to));
            let mut c = buf(&q, &[<$c as bonefish::Element>::ZERO; 4]);
            let no = Transpose::No;
            let r = bonefish::blas::$func(&q, no, no, 2, 2, 2, 2.0, &a, 2, &b, 2, 0.0, &mut c, 2);
            if ran(stringify!($func), r) {
                let got: Vec<f32> = c.to_vec().unwrap().into_iter().map(from).collect();
                assert_eq!(got, WANT.to_vec());
            }
        }
    };
}

blocking!(blocking_hgemm, hgemm, f16, f16, f16::from_f32, f16::to_f32);
blocking!(
    blocking_hgemm_f32,
    hgemm_f32,
    f16,
    f32,
    f16::from_f32,
    |x| x
);
blocking!(
    blocking_bgemm,
    bgemm,
    bf16,
    bf16,
    bf16::from_f32,
    bf16::to_f32
);
blocking!(
    blocking_bgemm_f32,
    bgemm_f32,
    bf16,
    f32,
    bf16::from_f32,
    |x| x
);

#[test]
fn blocking_rejects_foreign_queue_buffer() {
    common::arm();
    let Some(q) = queue("foreign") else { return };
    let q2 = Queue::new(DeviceKind::Cpu).unwrap();
    let a = buf(&q2, &A.map(f16::from_f32));
    let b = buf(&q, &B.map(f16::from_f32));
    let mut c = buf(&q, &[f16::ZERO; 4]);
    let no = Transpose::No;
    let r = bonefish::blas::hgemm(&q, no, no, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
    assert!(matches!(r, Err(Error::QueueMismatch)), "{r:?}");
}
