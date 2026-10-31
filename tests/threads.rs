//! Several host threads using the safe API. This tests the claim that `Queue` and `UsmBuffer` are
//! `Send + Sync`; it cannot prove it, and on the first Linux CI run it found a problem: with a
//! queue shared by four threads, the combined `gemm` test and the per-thread-DFT test both ran for
//! over 60 seconds (oneAPI 2026.2, OpenCL CPU, ubuntu runner) where Windows/2026.1 takes ~1.4 s.
//!
//! So each operation is isolated below, thread counts are varied, and the control that matters is
//! included: the same work with one queue **per thread** instead of one shared queue. Reading the
//! results: if the per-thread-queue control also hangs, the cause is concurrent oneMKL/OpenCL work
//! in general (oversubscription, a runtime lock), not queue sharing; if only the shared variants
//! hang, sharing a queue is the hazard.
//!
//! What it found (details in `ROADMAP.md` R3b): the per-thread-queue control fails too, so queue
//! sharing is not the cause. The cause is oneMKL itself: its first concurrent use is not thread-safe
//! in the shipped build (Windows, oneAPI 2026.1: crashes in `mkl_sycl_blas.6.dll`), contrary to the
//! oneAPI specification. The shim therefore holds one process-wide lock around every oneMKL call.
//! These tests are the regression guard for that: they start cold (the race is in first use, so
//! only a fresh process exercises it), and with `BONEFISH_UNSAFE_NO_MKL_LOCK=1` they fail on the
//! same machine, which is how the lock was shown to matter.
//!
//! A hang must be a named failure, not a 30-minute stall, so every concurrent run has a watchdog.
//! The tests are also serialised against each other: Rust's harness runs tests in parallel in one
//! process, and overlapping thread storms would make a result unattributable.
//!
//! Needs a SYCL device; without oneAPI these skip loudly unless `BONEFISH_REQUIRE_SYCL=1`.
//! CPU (OpenCL) device only; nothing here has run on a GPU.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bonefish::{
    Complex, DeviceKind, Dft, Placement, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer, UsmKind, blas,
};

/// How long a concurrent run may take before it is declared hung. Windows/2026.1 finishes the
/// heaviest of these in about a second.
const WATCHDOG: Duration = Duration::from_secs(45);

static SERIAL: Mutex<()> = Mutex::new(());

fn have_sycl(test: &str) -> bool {
    if !SYCL_AVAILABLE {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but bonefish was built without oneAPI"
        );
        eprintln!("SKIP {test}: built without oneAPI. Nothing ran on a device.");
        return false;
    }
    true
}

fn new_queue() -> Queue {
    Queue::new(DeviceKind::Cpu).expect("a CPU SYCL device")
}

/// Run `work(thread_index, queue)` on `threads` OS threads and wait for all of them, with a
/// watchdog. `shared`: every thread uses a clone of one queue; otherwise each creates its own.
/// A hang panics with the test name, how many threads finished and the variant, so the CI log says
/// which operation and which configuration stalled. The stalled threads are abandoned (they die
/// when the test process exits).
fn race(
    test: &str,
    threads: usize,
    shared: bool,
    work: impl Fn(usize, &Queue) + Send + Sync + 'static,
) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let work = Arc::new(work);
    let q = new_queue();
    let start = Instant::now();
    let (tx, rx) = mpsc::channel::<Result<(), String>>();
    for t in 0..threads {
        let (tx, work) = (tx.clone(), Arc::clone(&work));
        let q = if shared { Some(q.clone()) } else { None };
        std::thread::spawn(move || {
            let own;
            let qref = match &q {
                Some(q) => q,
                None => {
                    own = new_queue();
                    &own
                }
            };
            let r = catch_unwind(AssertUnwindSafe(|| work(t, qref))).map_err(|p| {
                p.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "worker panicked".into())
            });
            let _ = tx.send(r);
        });
    }
    drop(tx);
    let mut done = 0;
    while done < threads {
        let left = WATCHDOG.saturating_sub(start.elapsed());
        match rx.recv_timeout(left) {
            Ok(Ok(())) => done += 1,
            Ok(Err(msg)) => panic!("{test}: a worker failed: {msg}"),
            Err(_) => panic!(
                "{test}: HANG. {done}/{threads} threads finished after {WATCHDOG:?} \
                 (shared queue: {shared}). A stalled run, not a wrong answer."
            ),
        }
    }
    eprintln!(
        "{test}: {threads} threads, shared={shared}: finished in {:?}",
        start.elapsed()
    );
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

const ROUNDS: usize = 40;

// ---- one operation at a time ---------------------------------------------------------------

fn alloc_work(t: usize, q: &Queue) {
    for round in 0..ROUNDS {
        let data = [t as f64, round as f64, 3.0];
        let shared = UsmBuffer::from_slice(q, UsmKind::Shared, &data).unwrap();
        let device = UsmBuffer::<f64>::zeroed(q, UsmKind::Device, 3).unwrap();
        assert_eq!(shared.to_vec().unwrap(), data, "thread {t} round {round}");
        assert_eq!(device.to_vec().unwrap(), [0.0; 3]);
    }
}

fn memcpy_work(t: usize, q: &Queue) {
    let mut dev = UsmBuffer::<f64>::zeroed(q, UsmKind::Device, 64).unwrap();
    for round in 0..ROUNDS {
        let data: Vec<f64> = (0..64)
            .map(|i| (t * 1000 + round * 10 + i) as f64)
            .collect();
        dev.copy_from_slice(&data).unwrap();
        assert_eq!(dev.to_vec().unwrap(), data, "thread {t} round {round}");
    }
}

fn gemm_work(t: usize, q: &Queue) {
    // Each thread scales by its own factor, so a result that leaked between threads (wrong
    // buffer, wrong queue state) shows up as the wrong number.
    let scale = (t + 2) as f64;
    for round in 0..ROUNDS {
        // A = [1 3; 2 4], B = scale * I  =>  C = scale * A.
        let a = UsmBuffer::from_slice(q, UsmKind::Shared, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = UsmBuffer::from_slice(q, UsmKind::Shared, &[scale, 0.0, 0.0, scale]).unwrap();
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
}

/// One DFT plan per thread, none shared: the pattern the oneMKL specification allows for
/// class-based objects.
fn dft_work(t: usize, q: &Queue) {
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
}

macro_rules! race_test {
    ($name:ident, $threads:expr, $shared:expr, $work:ident) => {
        #[test]
        fn $name() {
            if have_sycl(stringify!($name)) {
                race(stringify!($name), $threads, $shared, $work);
            }
        }
    };
}

// Shared queue, the claim under test, four threads.
race_test!(shared_queue_alloc_only_4_threads, 4, true, alloc_work);
race_test!(shared_queue_memcpy_only_4_threads, 4, true, memcpy_work);
race_test!(shared_queue_gemm_4_threads, 4, true, gemm_work);
race_test!(
    shared_queue_dft_plan_per_thread_4_threads,
    4,
    true,
    dft_work
);

// Thread count: does a small count pass where four stall (oversubscription) or not?
race_test!(shared_queue_gemm_2_threads, 2, true, gemm_work);
race_test!(shared_queue_gemm_1_thread, 1, true, gemm_work);

// The control: one queue per thread. Decides what the shared-queue failures mean.
race_test!(own_queue_per_thread_gemm_4_threads, 4, false, gemm_work);
race_test!(own_queue_per_thread_dft_4_threads, 4, false, dft_work);

/// A buffer moved to another thread, and the queue dropped on yet another, must neither crash nor
/// leak the allocation: the buffer holds its queue alive.
#[test]
fn a_buffer_can_be_moved_to_another_thread_and_outlive_the_original_queue_handle() {
    if !have_sycl("buffer_moves_threads") {
        return;
    }
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let q = new_queue();
    let b = UsmBuffer::from_slice(&q, UsmKind::Shared, &[1.0_f32, 2.0, 3.0]).unwrap();
    drop(q);
    let got = std::thread::spawn(move || b.to_vec().unwrap())
        .join()
        .unwrap();
    assert_eq!(got, [1.0, 2.0, 3.0]);
}
