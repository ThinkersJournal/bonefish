//! Device-time profiling and the launch floor (R5b, `docs/FUEL-INTEL-GPU-DESIGN.md`). Needs a
//! device; without oneAPI the tests skip loudly unless `BONEFISH_REQUIRE_SYCL=1`. Measured on the
//! CPU/OpenCL device only: **no GPU has run any of this**, so whether a GPU queue supports
//! profiling, and what its timestamps mean, is UNVERIFIED.

use bonefish::{DeviceKind, Error, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer, UsmKind};

fn profiling_queue(test: &str) -> Option<Queue> {
    if !SYCL_AVAILABLE {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but bonefish was built without oneAPI"
        );
        eprintln!("SKIP {test}: built without oneAPI. Nothing ran on a device.");
        return None;
    }
    Some(Queue::new_profiling(DeviceKind::Cpu).expect("a CPU SYCL device"))
}

/// Run `f` on its own thread and fail with the test's name if it makes no progress for 60 s, so a
/// hang is a named failure instead of a silent stall (ROADMAP R8: this binary stalled once in about
/// 440 runs, cause unknown). The stuck thread is left behind.
fn watchdog(name: &str, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(60)) {
        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            handle.join().expect("test thread panicked");
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{name}: no progress in 60 s (hang)")
        }
    }
}

fn matrix(q: &Queue, n: usize, v: f32) -> UsmBuffer<f32> {
    UsmBuffer::from_slice(q, UsmKind::Shared, &vec![v; n * n]).unwrap()
}

#[test]
fn a_profiling_queue_reports_device_time_for_a_gemm() {
    watchdog("a_profiling_queue_reports_device_time_for_a_gemm", || {
        let Some(q) = profiling_queue("gemm_time") else {
            return;
        };
        assert!(q.is_profiling());
        let n = 512;
        let (a, b) = (matrix(&q, n, 1.0), matrix(&q, n, 1.0));
        let mut c = matrix(&q, n, 0.0);
        let t = q
            .scope(|s| {
                let (a, b) = (s.read(&a)?, s.read(&b)?);
                let c = s.write(&mut c)?;
                let e = s.gemm(
                    Transpose::No,
                    Transpose::No,
                    n,
                    n,
                    n,
                    1.0,
                    &a,
                    n,
                    &b,
                    n,
                    0.0,
                    &c,
                    n,
                )?;
                e.device_time()
            })
            .unwrap();
        assert!(t.start_ns > 0, "{t:?}");
        assert!(t.end_ns > t.start_ns, "no time elapsed: {t:?}");
        // The answer is right, so the time is the time of a real multiply.
        assert!(c.to_vec().unwrap().iter().all(|x| *x == n as f32));
    });
}

/// The same call on a queue made without profiling is an error that says so, never a zero time.
#[test]
fn a_queue_without_profiling_refuses_to_report_device_time() {
    watchdog(
        "a_queue_without_profiling_refuses_to_report_device_time",
        || {
            if profiling_queue("no_profiling").is_none() {
                return;
            }
            let plain = Queue::new(DeviceKind::Cpu).unwrap();
            assert!(!plain.is_profiling());
            let t = plain
                .scope(|s| {
                    let e = s.noop()?;
                    // Control inside the same scope: the event itself works, it is only the time that is refused.
                    e.wait()?;
                    Ok(e.device_time())
                })
                .unwrap();
            let e = t.unwrap_err();
            assert!(
                matches!(&e, Error::InvalidArgument(m) if m.contains("profiling")),
                "{e}"
            );
            // Control: the identical call on a profiling queue succeeds.
            let q = Queue::new_profiling(DeviceKind::Cpu).unwrap();
            let t = q
                .scope(|s| s.noop()?.device_time())
                .expect("a profiling queue reports the time of an empty kernel");
            assert!(t.end_ns >= t.start_ns);
        },
    );
}

/// The launch floor must be far below a real operation, or it is not a floor. Medians over 7
/// runs each; a 1024^2 f32 `gemm` against an empty kernel on the same queue.
#[test]
fn the_empty_kernel_is_faster_than_a_real_gemm() {
    watchdog("the_empty_kernel_is_faster_than_a_real_gemm", || {
        let Some(q) = profiling_queue("floor") else {
            return;
        };
        let n = 1024;
        let (a, b) = (matrix(&q, n, 1.0), matrix(&q, n, 1.0));
        let mut c = matrix(&q, n, 0.0);
        let (mut floors, mut gemms) = (Vec::new(), Vec::new());
        for _ in 0..7 {
            q.scope(|s| {
                let (a, b) = (s.read(&a)?, s.read(&b)?);
                let c = s.write(&mut c)?;
                let floor = s.noop()?;
                let gemm = s.gemm(
                    Transpose::No,
                    Transpose::No,
                    n,
                    n,
                    n,
                    1.0,
                    &a,
                    n,
                    &b,
                    n,
                    0.0,
                    &c,
                    n,
                )?;
                floors.push(floor.device_time()?.duration_ns());
                gemms.push(gemm.device_time()?.duration_ns());
                Ok(())
            })
            .unwrap();
        }
        floors.sort_unstable();
        gemms.sort_unstable();
        let (floor, gemm) = (floors[3], gemms[3]);
        eprintln!("median device time: empty kernel {floor} ns, gemm 1024^2 f32 {gemm} ns");
        assert!(floor > 0, "an empty kernel took no time at all");
        assert!(
            floor * 10 < gemm,
            "the floor ({floor} ns) is not far below a real gemm ({gemm} ns)"
        );
    });
}

/// A sanity check on the profiling timestamps: for two consecutive operations in one scope, the
/// second one starts no earlier than the first one ends on the device clock.
///
/// **This does not test the shim's submission chain.** It passes 25 of 25 runs with the chain
/// switched off (`BONEFISH_UNSAFE_NO_ASYNC_CHAIN=1`), because an in-order queue does order a native
/// empty kernel after a `gemm`; it is oneMKL's own `dot` that ignores the queue's order
/// (`docs/ASYNC-DESIGN.md` section 8). What it does check is that start/end are on one clock and
/// consistent with serial execution.
#[test]
fn device_timestamps_of_consecutive_operations_do_not_overlap() {
    watchdog(
        "device_timestamps_of_consecutive_operations_do_not_overlap",
        || {
            let Some(q) = profiling_queue("chain_times") else {
                return;
            };
            let n = 512;
            let (a, b) = (matrix(&q, n, 1.0), matrix(&q, n, 1.0));
            let mut c = matrix(&q, n, 0.0);
            for round in 0..10 {
                let (first, second) = q
                    .scope(|s| {
                        let (a, b) = (s.read(&a)?, s.read(&b)?);
                        let c = s.write(&mut c)?;
                        let e1 = s.gemm(
                            Transpose::No,
                            Transpose::No,
                            n,
                            n,
                            n,
                            1.0,
                            &a,
                            n,
                            &b,
                            n,
                            0.0,
                            &c,
                            n,
                        )?;
                        let e2 = s.noop()?;
                        Ok((e1.device_time()?, e2.device_time()?))
                    })
                    .unwrap();
                assert!(
                    second.start_ns >= first.end_ns,
                    "round {round}: the second operation started at {} before the first ended at {}",
                    second.start_ns,
                    first.end_ns
                );
            }
        },
    );
}
