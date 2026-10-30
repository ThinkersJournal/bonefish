//! A queue shared by several host threads. This exercises the claim that `Queue` and `UsmBuffer`
//! are `Send + Sync`; it does not prove it. The claim rests on the specification texts quoted in
//! `src/queue.rs`. Concurrency bugs are timing-dependent, so a pass here is evidence the common
//! case works, not that races are impossible.
//!
//! Needs a SYCL device; without oneAPI these skip loudly unless `BONEFISH_REQUIRE_SYCL=1`.
//! CPU (OpenCL) device only; nothing here has run on a GPU.

use bonefish::{
    Complex, DeviceKind, Dft, Placement, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer, UsmKind, blas,
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

/// The marker traits, checked at compile time in both builds (no device needed).
#[test]
fn queue_and_buffers_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Queue>();
    send_sync::<UsmBuffer<f32>>();
    send_sync::<UsmBuffer<f64>>();
    send_sync::<UsmBuffer<Complex<f32>>>();
    send_sync::<UsmBuffer<Complex<f64>>>();
    // `Dft` being neither is asserted by the `compile_fail,E0277` doctests on `Dft`.
}

const THREADS: usize = 4;
const ROUNDS: usize = 40;

#[test]
fn threads_share_one_queue_for_allocation_copies_and_gemm() {
    let Some(q) = queue("threads_share_one_queue") else {
        return;
    };
    std::thread::scope(|s| {
        for t in 0..THREADS {
            let q = &q;
            s.spawn(move || {
                // Each thread scales by its own factor, so a result that leaked between threads
                // (wrong buffer, wrong queue state) shows up as the wrong number.
                let scale = (t + 2) as f64;
                for round in 0..ROUNDS {
                    // A = [1 3; 2 4], B = scale * I  =>  C = scale * A.
                    let a =
                        UsmBuffer::from_slice(q, UsmKind::Shared, &[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let b = UsmBuffer::from_slice(q, UsmKind::Device, &[scale, 0.0, 0.0, scale])
                        .unwrap();
                    let mut c = UsmBuffer::<f64>::zeroed(q, UsmKind::Shared, 4).unwrap();
                    blas::gemm(
                        q,
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
                    .unwrap_or_else(|e| panic!("thread {t} round {round}: {e}"));
                    let want = [scale, 2.0 * scale, 3.0 * scale, 4.0 * scale];
                    assert_eq!(c.to_vec().unwrap(), want, "thread {t} round {round}");
                }
            });
        }
    });
}

/// The pattern the spec allows for class-based objects: one plan per thread, none shared.
#[test]
fn each_thread_builds_its_own_dft_plan_on_a_shared_queue() {
    let Some(q) = queue("each_thread_own_dft") else {
        return;
    };
    std::thread::scope(|s| {
        for t in 0..THREADS {
            let q = &q;
            s.spawn(move || {
                let plan = Dft::<f64>::c2c(q, &[8], Placement::InPlace, 1.0, 1.0).unwrap();
                for _ in 0..10 {
                    let mut data = vec![Complex::new(0.0, 0.0); 8];
                    data[0] = Complex::new(1.0 + t as f64, 0.0);
                    let mut x = UsmBuffer::from_slice(q, UsmKind::Shared, &data).unwrap();
                    plan.forward_in_place(&mut x).unwrap();
                    // The transform of a delta of height h is h everywhere.
                    let want = vec![Complex::new(1.0 + t as f64, 0.0); 8];
                    assert_eq!(x.to_vec().unwrap(), want, "thread {t}");
                }
            });
        }
    });
}

/// A buffer moved to another thread, and the queue dropped on yet another, must neither crash nor
/// leak the allocation: the buffer holds its queue alive.
#[test]
fn a_buffer_can_be_moved_to_another_thread_and_outlive_the_original_queue_handle() {
    let Some(q) = queue("buffer_moves_threads") else {
        return;
    };
    let b = UsmBuffer::from_slice(&q, UsmKind::Shared, &[1.0_f32, 2.0, 3.0]).unwrap();
    drop(q);
    let got = std::thread::spawn(move || b.to_vec().unwrap())
        .join()
        .unwrap();
    assert_eq!(got, [1.0, 2.0, 3.0]);
}
