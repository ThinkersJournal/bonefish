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

// ---- strided batched (hgemm_batch_strided and friends) ----

/// Column-major `c = alpha * op(a) * op(b) + beta * c` in f32 on the host: the reference.
#[allow(clippy::too_many_arguments)]
fn ref_gemm(
    ta: Transpose,
    tb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut s = 0.0;
            for l in 0..k {
                let av = if ta == Transpose::No {
                    a[i + l * lda]
                } else {
                    a[l + i * lda]
                };
                let bv = if tb == Transpose::No {
                    b[l + j * ldb]
                } else {
                    b[j + l * ldb]
                };
                s += av * bv;
            }
            c[i + j * ldc] = alpha * s + beta * c[i + j * ldc];
        }
    }
}

const GARBAGE: f32 = 77.0;

/// The property set in one batch: transposed A, B shared by stride 0, padded `lda`-stride and
/// padded `ldc`/`stride_c` (padding must stay untouched), alpha != 1, beta != 0 onto a preloaded C.
/// Every batch member differs, so a wrong stride or batch index changes the answer.
macro_rules! batch_variant {
    ($test:ident, $scope_m:ident, $block_f:ident, $a:ty, $c:ty, $conv:expr, $back:expr, $cconv:expr) => {
        #[test]
        fn $test() {
            common::arm();
            let Some(q) = queue(stringify!($test)) else {
                return;
            };
            let to: fn(f32) -> $a = $conv;
            let from: fn($c) -> f32 = $back;
            let cto: fn(f32) -> $c = $cconv;
            let (m, n, k, batch) = (2usize, 2usize, 3usize, 3usize);
            let (ta, tb) = (Transpose::Trans, Transpose::No);
            // op(A) is m x k, so A is k x m = 3 x 2, lda 3, stride 8 (2 padding elements).
            let (lda, sa) = (3usize, 8usize);
            let mut ah = vec![GARBAGE; sa * (batch - 1) + lda * m];
            for i in 0..batch {
                for e in 0..lda * m {
                    ah[i * sa + e] = (1 + (e + 2 * i) % 5) as f32;
                }
            }
            // B is k x n = 3 x 2, shared: stride 0.
            let bh: Vec<f32> = vec![1.0, 2.0, 1.0, 0.0, 1.0, 3.0];
            // C is m x n with ldc 3, stride 7 (>= one matrix: 2*3+... = (n-1)*ldc+m = 5).
            let (ldc, sc) = (3usize, 7usize);
            let mut c0 = vec![-1.0_f32; sc * (batch - 1) + ldc * n];
            for i in 0..batch {
                for j in 0..n {
                    for r in 0..m {
                        c0[i * sc + r + j * ldc] = (10 + i + r + j) as f32;
                    }
                }
            }
            let mut want = c0.clone();
            for i in 0..batch {
                ref_gemm(
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    2.0,
                    &ah[i * sa..],
                    lda,
                    &bh,
                    3,
                    1.0,
                    &mut want[i * sc..],
                    ldc,
                );
            }
            let a = buf(&q, &ah.iter().map(|x| to(*x)).collect::<Vec<$a>>());
            let b = buf(&q, &bh.iter().map(|x| to(*x)).collect::<Vec<$a>>());
            let c0v: Vec<$c> = c0.iter().map(|x| cto(*x)).collect();

            let mut c = buf(&q, &c0v);
            let r = q.scope(|s| {
                let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
                s.$scope_m(
                    ta, tb, m, n, k, 2.0, &a, lda, sa, &b, 3, 0, 1.0, &cw, ldc, sc, batch,
                )?;
                Ok(())
            });
            if ran(stringify!($scope_m), r) {
                let got: Vec<f32> = c.to_vec().unwrap().into_iter().map(from).collect();
                assert_eq!(got, want, "scope");
            }

            let mut c = buf(&q, &c0v);
            let r = bonefish::blas::$block_f(
                &q, ta, tb, m, n, k, 2.0, &a, lda, sa, &b, 3, 0, 1.0, &mut c, ldc, sc, batch,
            );
            if ran(stringify!($block_f), r) {
                let got: Vec<f32> = c.to_vec().unwrap().into_iter().map(from).collect();
                assert_eq!(got, want, "blocking");
            }
        }
    };
}

batch_variant!(
    hgemm_batch_strided_all_properties,
    hgemm_batch_strided,
    hgemm_batch_strided,
    f16,
    f16,
    f16::from_f32,
    f16::to_f32,
    f16::from_f32
);
batch_variant!(
    hgemm_f32_batch_strided_all_properties,
    hgemm_f32_batch_strided,
    hgemm_f32_batch_strided,
    f16,
    f32,
    f16::from_f32,
    |x| x,
    |x| x
);
batch_variant!(
    bgemm_batch_strided_all_properties,
    bgemm_batch_strided,
    bgemm_batch_strided,
    bf16,
    bf16,
    bf16::from_f32,
    bf16::to_f32,
    bf16::from_f32
);
batch_variant!(
    bgemm_f32_batch_strided_all_properties,
    bgemm_f32_batch_strided,
    bgemm_f32_batch_strided,
    bf16,
    f32,
    bf16::from_f32,
    |x| x,
    |x| x
);

/// Accumulation precision of the f32-output variants: a dot of 3000 ones is 3000 exactly in f32
/// but is not representable in f16 past 2048 (nor in bf16 past 256), so a narrow accumulator
/// cannot produce it.
macro_rules! accumulation {
    ($test:ident, $scope_m:ident, $a:ty, $conv:expr) => {
        #[test]
        fn $test() {
            common::arm();
            let Some(q) = queue(stringify!($test)) else {
                return;
            };
            let to: fn(f32) -> $a = $conv;
            let k = 3000usize;
            let ones = vec![to(1.0); k * 2];
            let a = buf(&q, &ones); // 1 x k, two batch members
            let b = buf(&q, &ones); // k x 1
            let mut c = buf(&q, &[0.0_f32; 2]);
            let no = Transpose::No;
            let r = q.scope(|s| {
                let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
                s.$scope_m(no, no, 1, 1, k, 1.0, &a, 1, k, &b, k, k, 0.0, &cw, 1, 1, 2)?;
                Ok(())
            });
            if ran(stringify!($scope_m), r) {
                assert_eq!(c.to_vec().unwrap(), vec![3000.0, 3000.0]);
            }
        }
    };
}
accumulation!(
    hgemm_f32_batch_accumulates_in_f32,
    hgemm_f32_batch_strided,
    f16,
    f16::from_f32
);
accumulation!(
    bgemm_f32_batch_accumulates_in_f32,
    bgemm_f32_batch_strided,
    bf16,
    bf16::from_f32
);

#[test]
fn batch_extents_strides_and_aliasing_are_checked() {
    common::arm();
    let Some(q) = queue("batch_checks") else {
        return;
    };
    let a = buf(&q, &[f16::ONE; 8]);
    let b = buf(&q, &[f16::ONE; 4]);
    let mut c = buf(&q, &[f16::ZERO; 8]);
    let no = Transpose::No;
    let bad = |r: Result<(), Error>, what: &str| {
        assert!(matches!(r, Err(Error::InvalidArgument(_))), "{what}: {r:?}");
    };
    // Positive control: 2 x (2x2) at stride 4 fits 8-element buffers.
    let ok = q.scope(|s| {
        let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
        s.hgemm_batch_strided(no, no, 2, 2, 2, 1.0, &a, 2, 4, &b, 2, 0, 0.0, &cw, 2, 4, 2)?;
        Ok(())
    });
    let _ = ran("batch_checks positive control", ok);
    // Batch of 3 at stride 4 needs 12 elements of A.
    bad(
        q.scope(|s| {
            let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
            s.hgemm_batch_strided(no, no, 2, 2, 2, 1.0, &a, 2, 4, &b, 2, 0, 0.0, &cw, 2, 4, 3)?;
            Ok(())
        }),
        "batch beyond A",
    );
    // stride_c 2 is below one C matrix (4), so outputs would overlap.
    bad(
        q.scope(|s| {
            let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
            s.hgemm_batch_strided(no, no, 2, 2, 2, 1.0, &a, 2, 4, &b, 2, 0, 0.0, &cw, 2, 2, 2)?;
            Ok(())
        }),
        "overlapping C",
    );
    // lda below the row count.
    bad(
        q.scope(|s| {
            let (a, b, cw) = (s.read(&a)?, s.read(&b)?, s.write(&mut c)?);
            s.hgemm_batch_strided(no, no, 2, 2, 2, 1.0, &a, 1, 4, &b, 2, 0, 0.0, &cw, 2, 4, 2)?;
            Ok(())
        }),
        "lda",
    );
    // Output is also an input.
    bad(
        q.scope(|s| {
            let b2 = s.read(&b)?;
            let a2 = s.write(&mut c)?;
            s.hgemm_batch_strided(
                no, no, 2, 2, 2, 1.0, &a2, 2, 4, &b2, 2, 0, 0.0, &a2, 2, 4, 2,
            )?;
            Ok(())
        }),
        "aliasing",
    );
    // Blocking: foreign queue.
    let q2 = Queue::new(DeviceKind::Cpu).unwrap();
    let a2 = buf(&q2, &[f16::ONE; 8]);
    let r = bonefish::blas::hgemm_batch_strided(
        &q, no, no, 2, 2, 2, 1.0, &a2, 2, 4, &b, 2, 0, 0.0, &mut c, 2, 4, 2,
    );
    assert!(matches!(r, Err(Error::QueueMismatch)), "{r:?}");
}
