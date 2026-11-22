//! Strided batched `gemm` through the safe API (blocking and on `Scope`), hand values over a batch
//! of 2, plus the extent and stride errors. Needs a device; without oneAPI the tests skip loudly
//! unless `BONEFISH_REQUIRE_SYCL=1`. CPU/OpenCL device only; GPU UNVERIFIED.

use bonefish::blas::gemm_batch_strided;
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

// Column-major. Batch 0: A0 = [[1,2],[3,4]], B0 = I. Batch 1: A1 = 2 I, B1 = [[1,2],[3,4]].
// Expected C0 = A0, C1 = 2 * B1 = [[2,4],[6,8]].
const A: [f64; 8] = [1.0, 3.0, 2.0, 4.0, 2.0, 0.0, 0.0, 2.0];
const B: [f64; 8] = [1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 4.0];
const WANT: [f64; 8] = [1.0, 3.0, 2.0, 4.0, 2.0, 6.0, 4.0, 8.0];

#[test]
fn blocking_f64() {
    let Some(q) = queue("blocking_f64") else {
        return;
    };
    let (a, b, mut c) = (buf(&q, &A), buf(&q, &B), buf(&q, &[0.0f64; 8]));
    let no = Transpose::No;
    gemm_batch_strided(&q, no, no, 2, 2, 2, 1.0, &a, 2, 4, &b, 2, 4, 0.0, &mut c, 2, 4, 2).unwrap();
    assert_eq!(c.to_vec().unwrap(), WANT);
}

#[test]
fn blocking_f32() {
    let Some(q) = queue("blocking_f32") else {
        return;
    };
    let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let (a, b, mut c) = (buf(&q, &f(&A)), buf(&q, &f(&B)), buf(&q, &[0.0f32; 8]));
    let no = Transpose::No;
    gemm_batch_strided(&q, no, no, 2, 2, 2, 1.0f32, &a, 2, 4, &b, 2, 4, 0.0, &mut c, 2, 4, 2)
        .unwrap();
    assert_eq!(c.to_vec().unwrap(), f(&WANT));
}

#[test]
fn stride_zero_reuses_one_matrix() {
    let Some(q) = queue("stride_zero") else {
        return;
    };
    // A0 = [[1,2],[3,4]] for both products; B = [I | 2I]; C = [A0 | 2 A0].
    let a = buf(&q, &A[..4]);
    let b = buf(&q, &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let mut c = buf(&q, &[0.0f64; 8]);
    let no = Transpose::No;
    gemm_batch_strided(&q, no, no, 2, 2, 2, 1.0, &a, 2, 0, &b, 2, 4, 0.0, &mut c, 2, 4, 2).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![1.0, 3.0, 2.0, 4.0, 2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn on_scope_matches_blocking() {
    let Some(q) = queue("on_scope") else { return };
    let (a, b, mut c) = (buf(&q, &A), buf(&q, &B), buf(&q, &[0.0f64; 8]));
    let no = Transpose::No;
    q.scope(|s| {
        let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
        s.gemm_batch_strided(no, no, 2, 2, 2, 1.0, &a, 2, 4, &b, 2, 4, 0.0, &cw, 2, 4, 2)?;
        Ok(())
    })
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), WANT);
}

#[test]
fn extents_and_strides_are_checked() {
    let Some(q) = queue("checks") else { return };
    let (a, b, mut c) = (buf(&q, &A), buf(&q, &B), buf(&q, &[0.0f64; 8]));
    let no = Transpose::No;
    // The whole batch must fit: 3 matrices at stride 4 need 12 elements.
    let r = gemm_batch_strided(&q, no, no, 2, 2, 2, 1.0, &a, 2, 4, &b, 2, 4, 0.0, &mut c, 2, 4, 3);
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
    // Outputs must not overlap: stride_c 2 is below one 2 x 2 matrix.
    let r = gemm_batch_strided(&q, no, no, 2, 2, 2, 1.0, &a, 2, 4, &b, 2, 4, 0.0, &mut c, 2, 2, 2);
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
    // Same-buffer output is refused (through the scope form, which sees the aliasing).
    let r = q.scope(|s| {
        let b2 = s.read(&b)?;
        let a2 = s.write(&mut c)?;
        s.gemm_batch_strided(no, no, 2, 2, 2, 1.0, &a2, 2, 4, &b2, 2, 4, 0.0, &a2, 2, 4, 2)?;
        Ok(())
    });
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
    // Nothing ran: C is untouched.
    assert_eq!(c.to_vec().unwrap(), vec![0.0; 8]);
}
