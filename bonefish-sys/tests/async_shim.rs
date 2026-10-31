//! The non-blocking shim layer: in-order queues, `*_async` calls returning events, waiting, and
//! asynchronous errors (docs/ASYNC-DESIGN.md sections 3.4, 4 and 5). CPU/OpenCL device only; GPU
//! behaviour is UNVERIFIED. Needs oneAPI at build time, so the file is empty without it.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};
use std::sync::Mutex;

// Async errors are per-thread and queue state is process-wide in the runtime; run these one at a
// time so one test's in-flight work cannot colour another's.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run `f` on its own thread and fail with the test's name if it takes longer than `secs`, so a
/// hang is a named failure instead of a silent stall (see ROADMAP R8). The stuck thread is left
/// behind; the process ends when the harness does.
fn watchdog(name: &str, secs: u64, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
        Ok(()) => handle.join().expect("test thread panicked"),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            // The closure panicked before sending: surface its panic.
            handle.join().expect("test thread panicked");
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{name}: no progress in {secs} s (hang)")
        }
    }
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(bf_last_error()) }
        .to_string_lossy()
        .into_owned()
}

struct Queue(*mut c_void);
impl Queue {
    fn new() -> Self {
        let q = unsafe { bf_queue_new(1) };
        assert!(!q.is_null(), "no CPU SYCL device: {}", last_error());
        Queue(q)
    }
}

/// Shared USM of `n` f32, zero-initialised. Declared after its queue so it drops first.
struct Buf<'q> {
    q: &'q Queue,
    p: *mut f32,
    n: usize,
}
impl<'q> Buf<'q> {
    fn new(q: &'q Queue, n: usize) -> Self {
        let p = unsafe { bf_malloc(q.0, 0, (n * 4) as i64) }.cast::<f32>();
        assert!(!p.is_null());
        unsafe { std::ptr::write_bytes(p, 0, n) };
        Buf { q, p, n }
    }
    fn fill(&self, f: impl Fn(usize) -> f32) {
        for i in 0..self.n {
            unsafe { *self.p.add(i) = f(i) };
        }
    }
    fn get(&self, i: usize) -> f32 {
        unsafe { *self.p.add(i) }
    }
}
impl Drop for Buf<'_> {
    fn drop(&mut self) {
        unsafe { bf_free(self.q.0, self.p.cast()) }
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { bf_queue_free(self.0) }
    }
}

/// `c = a * b` (n x n, column-major), non-blocking, after `deps`.
fn gemm_async(q: &Queue, n: i64, a: &Buf, b: &Buf, c: &Buf, deps: &[*mut c_void]) -> *mut c_void {
    unsafe {
        bf_sgemm_async(
            q.0,
            0,
            0,
            n,
            n,
            n,
            1.0,
            a.p,
            n,
            b.p,
            n,
            0.0,
            c.p,
            n,
            if deps.is_empty() {
                std::ptr::null()
            } else {
                deps.as_ptr()
            },
            deps.len() as i64,
        )
    }
}

#[test]
fn queues_are_in_order() {
    watchdog("queues_are_in_order", 60, || {
        let _g = serial();
        let q = Queue::new();
        assert_eq!(unsafe { bf_queue_is_in_order(q.0) }, 1);
    });
}

#[test]
fn async_gemm_result_arrives_after_wait() {
    watchdog("async_gemm_result_arrives_after_wait", 60, || {
        let _g = serial();
        let q = Queue::new();
        let (a, b, c) = (Buf::new(&q, 4), Buf::new(&q, 4), Buf::new(&q, 4));
        // column-major [[1,3],[2,4]] * 3I
        a.fill(|i| (i + 1) as f32);
        b.fill(|i| if i == 0 || i == 3 { 3.0 } else { 0.0 });
        let e = gemm_async(&q, 2, &a, &b, &c, &[]);
        assert!(!e.is_null(), "submit failed: {}", last_error());
        assert_eq!(unsafe { bf_event_wait(e) }, 0, "{}", last_error());
        unsafe { bf_event_free(e) };
        assert_eq!(
            (c.get(0), c.get(1), c.get(2), c.get(3)),
            (3.0, 6.0, 9.0, 12.0)
        );
    });
}

/// A chain of dependent products on one in-order queue with NO explicit dependencies: each step
/// reads what the previous wrote, so a wrong order shows as a wrong power of two.
///
/// **This test has no teeth on the CPU/OpenCL device, and `gemm` is not the operation that
/// matters.** With the `in_order` property removed from the shim, an out-of-order queue still
/// passed it 15 of 15 runs; and `gemm` after `gemm` is one of the pairs oneMKL orders by itself.
/// The ordering that the scoped API's soundness needs is pinned by
/// `asynchronous_calls_are_ordered_without_explicit_dependencies` (which includes `dot`, the call
/// that ignores an in-order queue) and `queues_are_in_order` pins only the property. GPU UNVERIFIED.
#[test]
fn in_order_queue_runs_a_dependent_chain_without_explicit_deps() {
    watchdog(
        "in_order_queue_runs_a_dependent_chain_without_explicit_deps",
        60,
        || {
            let _g = serial();
            let q = Queue::new();
            let n = 128usize;
            let (a, x, y) = (
                Buf::new(&q, n * n),
                Buf::new(&q, n * n),
                Buf::new(&q, n * n),
            );
            a.fill(|i| if i % (n + 1) == 0 { 2.0 } else { 0.0 }); // 2 * identity
            x.fill(|_| 1.0);
            let steps = 10;
            let mut events = Vec::new();
            for s in 0..steps {
                let (src, dst) = if s % 2 == 0 { (&x, &y) } else { (&y, &x) };
                let e = gemm_async(&q, n as i64, &a, src, dst, &[]);
                assert!(!e.is_null(), "step {s}: {}", last_error());
                events.push(e);
            }
            assert_eq!(unsafe { bf_queue_wait(q.0) }, 0, "{}", last_error());
            for e in events {
                unsafe { bf_event_free(e) };
            }
            let result = if steps % 2 == 0 { &x } else { &y };
            let want = (1u32 << steps) as f32;
            for i in [0, 1, n * n / 2, n * n - 1] {
                assert_eq!(result.get(i), want, "element {i}");
            }
        },
    );
}

/// Explicit dependency array: the second product waits on the first event.
#[test]
fn explicit_dependency_event_is_accepted() {
    watchdog("explicit_dependency_event_is_accepted", 60, || {
        let _g = serial();
        let q = Queue::new();
        let n = 64usize;
        let (a, x, y, z) = (
            Buf::new(&q, n * n),
            Buf::new(&q, n * n),
            Buf::new(&q, n * n),
            Buf::new(&q, n * n),
        );
        a.fill(|i| if i % (n + 1) == 0 { 2.0 } else { 0.0 });
        x.fill(|_| 1.0);
        let e1 = gemm_async(&q, n as i64, &a, &x, &y, &[]);
        assert!(!e1.is_null());
        let e2 = gemm_async(&q, n as i64, &a, &y, &z, &[e1]);
        assert!(!e2.is_null(), "{}", last_error());
        assert_eq!(unsafe { bf_event_wait(e2) }, 0, "{}", last_error());
        unsafe {
            bf_event_free(e1);
            bf_event_free(e2);
        }
        assert_eq!(z.get(0), 4.0);
        assert_eq!(z.get(n * n - 1), 4.0);
    });
}

/// A failure raised after submission must come back as a status, not abort the process. The
/// failure is forced with a host task that throws. Controls: the same wait calls return 0 when
/// nothing failed (previous tests), and a second wait after the failure is clean again.
#[test]
fn asynchronous_failure_surfaces_as_a_status_via_event_wait() {
    watchdog(
        "asynchronous_failure_surfaces_as_a_status_via_event_wait",
        60,
        || {
            let _g = serial();
            let q = Queue::new();
            let e = unsafe { bf_test_async_failure(q.0) };
            assert!(!e.is_null(), "{}", last_error());
            let rc = unsafe { bf_event_wait(e) };
            assert_eq!(rc, 1, "expected an asynchronous error status, got {rc}");
            let msg = last_error();
            assert!(
                msg.contains("forced asynchronous failure"),
                "message lost: {msg:?}"
            );
            unsafe { bf_event_free(e) };
            assert_eq!(
                unsafe { bf_queue_wait(q.0) },
                0,
                "failure was reported twice"
            );
        },
    );
}

#[test]
fn asynchronous_failure_surfaces_as_a_status_via_queue_wait() {
    watchdog(
        "asynchronous_failure_surfaces_as_a_status_via_queue_wait",
        60,
        || {
            let _g = serial();
            let q = Queue::new();
            let e = unsafe { bf_test_async_failure(q.0) };
            assert!(!e.is_null(), "{}", last_error());
            let rc = unsafe { bf_queue_wait(q.0) };
            assert_eq!(rc, 1, "expected an asynchronous error status, got {rc}");
            assert!(last_error().contains("forced asynchronous failure"));
            unsafe { bf_event_free(e) };
            assert_eq!(
                unsafe { bf_queue_wait(q.0) },
                0,
                "failure was reported twice"
            );
        },
    );
}

#[test]
fn waiting_on_a_null_event_is_an_error_not_a_crash() {
    let rc = unsafe { bf_event_wait(std::ptr::null_mut()) };
    assert_eq!(rc, 3);
}

/// Is an asynchronous error still reported when the caller drops the event before waiting on the
/// queue? The scoped safe API drops events freely, so its "the scope returns the first
/// asynchronous error" promise depends on the answer. Measured per variant over many fresh
/// queues; the summary line is printed so a CI log shows which variant lost errors.
///   A: event kept until after `bf_queue_wait`
///   B: event freed right after submission, then `bf_queue_wait`
///   C: as B, with a pause before the wait (separates "lost with the event" from "not yet posted")
#[test]
fn asynchronous_error_survives_dropping_the_event_before_the_queue_wait() {
    watchdog(
        "asynchronous_error_survives_dropping_the_event",
        120,
        || {
            let _g = serial();
            const N: usize = 100;
            let mut lost = [0usize; 3];
            for i in 0..N {
                for (variant, slot) in lost.iter_mut().enumerate() {
                    let q = Queue::new();
                    let e = unsafe { bf_test_async_failure(q.0) };
                    assert!(!e.is_null(), "{}", last_error());
                    if variant != 0 {
                        unsafe { bf_event_free(e) };
                    }
                    if variant == 2 {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    let rc = unsafe { bf_queue_wait(q.0) };
                    if variant == 0 {
                        unsafe { bf_event_free(e) };
                    }
                    if rc != 1 {
                        *slot += 1;
                    }
                    let _ = i;
                }
            }
            eprintln!(
                "ASYNC-ERROR-LOSS over {N} fresh queues: A(event kept)={} B(event freed)={} C(freed+50ms)={}",
                lost[0], lost[1], lost[2]
            );
            assert_eq!(
                lost,
                [0, 0, 0],
                "lost errors out of {N}: A(event kept)={} B(event freed)={} C(freed, 50 ms pause)={}",
                lost[0],
                lost[1],
                lost[2]
            );
        },
    );
}

/// The shape that lost an error once on Linux CI (the in-crate `scope` test): the queue has
/// already been waited on once, then a failing host task is submitted, its event freed, and the
/// queue waited on, repeatedly on one reused queue with nothing else running. Every wait must
/// report. (With other threads active the first wait can come back early; see
/// `a_missed_asynchronous_error_is_reported_by_a_later_wait`.)
#[test]
fn asynchronous_error_is_not_lost_on_a_reused_queue() {
    watchdog("asynchronous_error_reused_queue", 120, || {
        let _g = serial();
        const N: usize = 150;
        let q = Queue::new();
        let mut lost = 0usize;
        for _ in 0..N {
            // The control wait must be clean.
            assert_eq!(unsafe { bf_queue_wait(q.0) }, 0, "{}", last_error());
            let e = unsafe { bf_test_async_failure(q.0) };
            assert!(!e.is_null());
            unsafe { bf_event_free(e) };
            if unsafe { bf_queue_wait(q.0) } != 1 {
                lost += 1;
            }
        }
        assert_eq!(
            lost, 0,
            "{lost} of {N} sequential failures were not reported"
        );
    });
}

/// Errors must be attributed to the queue that failed. Two threads fail on their own queues while
/// two other threads wait on queues that never fail; a clean queue must never report an error.
/// (With a per-thread error list this reported foreign errors and lost about half of the real
/// ones; errors are now kept per queue.)
#[test]
fn a_clean_queue_never_reports_another_queues_asynchronous_error() {
    watchdog("clean_queue_never_reports_foreign_error", 120, || {
        let _g = serial();
        const N: usize = 150;
        let handles: Vec<_> = (0..4)
            .map(|t| {
                std::thread::spawn(move || {
                    let q = Queue::new();
                    let failing = t < 2;
                    let mut wrong = 0usize;
                    for _ in 0..N {
                        if failing {
                            let e = unsafe { bf_test_async_failure(q.0) };
                            assert!(!e.is_null());
                            unsafe { bf_event_free(e) };
                        }
                        let rc = unsafe { bf_queue_wait(q.0) };
                        // A failing queue must report; a clean queue must not.
                        if (rc == 1) != failing {
                            wrong += 1;
                        }
                    }
                    (failing, wrong)
                })
            })
            .collect();
        let (mut lost, mut foreign) = (0usize, 0usize);
        for h in handles {
            let (failing, wrong) = h.join().unwrap();
            if failing {
                lost += wrong;
            } else {
                foreign += wrong;
            }
        }
        // `lost` is informational: a failing queue's first wait can return before the runtime has
        // posted the error (a later wait reports it, see the next test). A clean queue reporting
        // an error is a wrong attribution and must never happen.
        eprintln!(
            "ASYNC-ERROR-ATTRIBUTION failing queues' first wait missed={lost}; clean queues reported a foreign error={foreign} (of {N} per thread)"
        );
        assert_eq!(
            foreign, 0,
            "a clean queue reported another queue's asynchronous error {foreign} times"
        );
    });
}

/// Classifies a missed error: when the first `bf_queue_wait` after a failure returns 0, is the
/// error gone, or does a later wait on the same queue still report it? Four threads, each on its
/// own queue, failing and freeing the event as above. `missed` counts first waits that returned
/// 0; `recovered` counts those where polling the same queue for up to 2 s then returned 1.
#[test]
fn a_missed_asynchronous_error_is_reported_by_a_later_wait() {
    watchdog("missed_error_is_reported_later", 240, || {
        let _g = serial();
        const N: usize = 150;
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(move || {
                    let q = Queue::new();
                    let (mut missed, mut recovered) = (0usize, 0usize);
                    for _ in 0..N {
                        let e = unsafe { bf_test_async_failure(q.0) };
                        assert!(!e.is_null());
                        unsafe { bf_event_free(e) };
                        if unsafe { bf_queue_wait(q.0) } == 1 {
                            continue;
                        }
                        missed += 1;
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(2);
                        while std::time::Instant::now() < deadline {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                            if unsafe { bf_queue_wait(q.0) } == 1 {
                                recovered += 1;
                                break;
                            }
                        }
                    }
                    (missed, recovered)
                })
            })
            .collect();
        let (mut missed, mut recovered) = (0usize, 0usize);
        for h in handles {
            let (m, r) = h.join().unwrap();
            missed += m;
            recovered += r;
        }
        eprintln!(
            "ASYNC-ERROR-LATE first wait missed={missed} later wait recovered={recovered} (of {} failures)",
            4 * N
        );
        assert_eq!(
            missed,
            recovered,
            "of {} failures, {missed} were missed by the first wait and only {recovered} were reported later",
            4 * N
        );
    });
}

/// Typed shared USM for the level-1 / gemv / DFT tests below. Declared after its queue.
struct Mem<'q, T: Copy> {
    q: &'q Queue,
    p: *mut T,
}
impl<'q, T: Copy> Mem<'q, T> {
    fn new(q: &'q Queue, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data).max(1) as i64;
        let p = unsafe { bf_malloc(q.0, 0, bytes) }.cast::<T>();
        assert!(!p.is_null());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
        Mem { q, p }
    }
    fn get(&self, n: usize) -> Vec<T> {
        (0..n).map(|i| unsafe { *self.p.add(i) }).collect()
    }
}
impl<T: Copy> Drop for Mem<'_, T> {
    fn drop(&mut self) {
        unsafe { bf_free(self.q.0, self.p.cast()) }
    }
}

const NO_DEPS: *const *mut c_void = std::ptr::null();

fn free_all(events: &[*mut c_void]) {
    for e in events {
        unsafe { bf_event_free(*e) };
    }
}

/// Every non-blocking level-1 function and gemv, submitted back to back on the in-order queue
/// without waiting in between, then one queue wait, then the results are read through the host
/// pointers. Hand-computed expectations (double precision).
#[test]
fn async_level_1_and_gemv_give_the_right_numbers() {
    watchdog("async_level_1_and_gemv", 60, || {
        let _g = serial();
        let q = Queue::new();
        let x = Mem::new(&q, &[1.0_f64, 2.0, 3.0, 4.0]);
        let y = Mem::new(&q, &[1.0_f64, 1.0, 1.0, 1.0]);
        let r_dot = Mem::new(&q, &[0.0_f64]);
        let r_nrm = Mem::new(&q, &[0.0_f64]);
        let r_asum = Mem::new(&q, &[0.0_f64]);
        let r_iamax = Mem::new(&q, &[-1_i64]);
        let mut ev = Vec::new();
        unsafe {
            // y = 2x + y = [3, 5, 7, 9]
            ev.push(bf_daxpy_async(q.0, 4, 2.0, x.p, 1, y.p, 1, NO_DEPS, 0));
            // dot(x, y) = 3 + 10 + 21 + 36 = 70, nrm2(x) = sqrt(30), asum(x) = 10, iamax(y) = 3
            ev.push(bf_ddot_async(q.0, 4, x.p, 1, y.p, 1, r_dot.p, NO_DEPS, 0));
            ev.push(bf_dnrm2_async(q.0, 4, x.p, 1, r_nrm.p, NO_DEPS, 0));
            ev.push(bf_dasum_async(q.0, 4, x.p, 1, r_asum.p, NO_DEPS, 0));
            ev.push(bf_diamax_async(q.0, 4, y.p, 1, r_iamax.p, NO_DEPS, 0));
        }
        assert!(ev.iter().all(|e| !e.is_null()), "{}", last_error());
        assert_eq!(unsafe { bf_queue_wait(q.0) }, 0, "{}", last_error());
        assert_eq!(y.get(4), vec![3.0, 5.0, 7.0, 9.0]);
        assert_eq!(r_dot.get(1), vec![70.0]);
        assert!((r_nrm.get(1)[0] - 30.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(r_asum.get(1), vec![10.0]);
        assert_eq!(r_iamax.get(1), vec![3]); // zero-based
        free_all(&ev);

        // scal, copy, swap, then gemv: x = 3x = [3, 6, 9, 12]; z = copy of x; swap(x, y).
        let z = Mem::new(&q, &[0.0_f64; 4]);
        let mut ev = Vec::new();
        unsafe {
            ev.push(bf_dscal_async(q.0, 4, 3.0, x.p, 1, NO_DEPS, 0));
            ev.push(bf_dcopy_async(q.0, 4, x.p, 1, z.p, 1, NO_DEPS, 0));
            ev.push(bf_dswap_async(q.0, 4, x.p, 1, y.p, 1, NO_DEPS, 0));
        }
        assert!(ev.iter().all(|e| !e.is_null()), "{}", last_error());
        assert_eq!(unsafe { bf_queue_wait(q.0) }, 0, "{}", last_error());
        assert_eq!(z.get(4), vec![3.0, 6.0, 9.0, 12.0]);
        assert_eq!(x.get(4), vec![3.0, 5.0, 7.0, 9.0]); // was y
        assert_eq!(y.get(4), vec![3.0, 6.0, 9.0, 12.0]); // was 3x
        free_all(&ev);

        // gemv, column-major A = [[1, 3], [2, 4]] (columns [1,2] and [3,4]), x = [1, 1]:
        // N: y = A x = [4, 6]; T: y = A^T x = [3, 7].
        let a = Mem::new(&q, &[1.0_f64, 2.0, 3.0, 4.0]);
        let v = Mem::new(&q, &[1.0_f64, 1.0]);
        let yn = Mem::new(&q, &[0.0_f64; 2]);
        let yt = Mem::new(&q, &[0.0_f64; 2]);
        let mut ev = Vec::new();
        unsafe {
            ev.push(bf_dgemv_async(
                q.0, 0, 2, 2, 1.0, a.p, 2, v.p, 1, 0.0, yn.p, 1, NO_DEPS, 0,
            ));
            ev.push(bf_dgemv_async(
                q.0, 1, 2, 2, 1.0, a.p, 2, v.p, 1, 0.0, yt.p, 1, NO_DEPS, 0,
            ));
        }
        assert!(ev.iter().all(|e| !e.is_null()), "{}", last_error());
        assert_eq!(unsafe { bf_queue_wait(q.0) }, 0, "{}", last_error());
        assert_eq!(yn.get(2), vec![4.0, 6.0]);
        assert_eq!(yt.get(2), vec![3.0, 7.0]);
        free_all(&ev);
    });
}

/// Single precision takes the same path; one operation each is enough to tell the macros apart.
#[test]
fn async_single_precision_level_1() {
    watchdog("async_single_precision_level_1", 60, || {
        let _g = serial();
        let q = Queue::new();
        let x = Mem::new(&q, &[1.0_f32, 2.0, 3.0]);
        let y = Mem::new(&q, &[10.0_f32, 20.0, 30.0]);
        let r = Mem::new(&q, &[0.0_f32]);
        let e1 = unsafe { bf_saxpy_async(q.0, 3, 2.0, x.p, 1, y.p, 1, NO_DEPS, 0) };
        // dot waits for the axpy through an explicit dependency array as well.
        let deps = [e1];
        let e2 = unsafe { bf_sdot_async(q.0, 3, x.p, 1, y.p, 1, r.p, deps.as_ptr(), 1) };
        assert!(!e1.is_null() && !e2.is_null(), "{}", last_error());
        assert_eq!(unsafe { bf_event_wait(e2) }, 0, "{}", last_error());
        assert_eq!(y.get(3), vec![12.0, 24.0, 36.0]);
        assert_eq!(r.get(1), vec![12.0 + 48.0 + 108.0]);
        free_all(&[e1, e2]);
    });
}

/// Asynchronous DFT: a delta transforms to all ones, in place, double precision.
#[test]
fn async_dft_compute_transforms_a_delta() {
    watchdog("async_dft_compute", 60, || {
        let _g = serial();
        let q = Queue::new();
        let dims = [4_i64];
        let h = unsafe { bf_dft_c2c_new(q.0, 1, 1, dims.as_ptr(), 1, 1.0, 1.0) };
        assert!(!h.is_null(), "{}", last_error());
        // Interleaved (re, im): [1, 0, 0, 0] as complex.
        let data = Mem::new(&q, &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let e =
            unsafe { bf_dft_compute_async(h, 0, data.p.cast(), std::ptr::null_mut(), NO_DEPS, 0) };
        assert!(!e.is_null(), "{}", last_error());
        assert_eq!(unsafe { bf_event_wait(e) }, 0, "{}", last_error());
        let out = data.get(8);
        for k in 0..4 {
            assert!(
                (out[2 * k] - 1.0).abs() < 1e-12 && out[2 * k + 1].abs() < 1e-12,
                "{out:?}"
            );
        }
        free_all(&[e]);
        unsafe { bf_dft_free(h) };
    });
}

/// Back-to-back oneMKL calls whose second reads what the first wrote, with no explicit dependency.
/// Measured (Windows, oneAPI 2026.1, CPU/OpenCL, in-order queue) with the shim's chain OFF: `dot`
/// submitted right after `axpy`/`copy`/`gemm` read the old data in 299/300, 199/200 and 200/200
/// rounds, and 0/200 when it listed the earlier event as a dependency; the other pairs below were
/// ordered. So an in-order queue alone does not order oneMKL calls. With the chain ON (the
/// default) every pair must be right. The OFF arm only reports (it is also the retest for a newer
/// oneMKL: if it reports 0 wrong everywhere, the chain may no longer be needed).
#[test]
fn asynchronous_calls_are_ordered_without_explicit_dependencies() {
    watchdog("asynchronous_calls_are_ordered", 180, || {
        let _g = serial();
        const N: usize = 100;
        const PAIRS: [&str; 8] = [
            "scal->axpy",
            "copy->dot",
            "axpy->dot",
            "axpy->asum",
            "axpy->nrm2",
            "gemm->gemm",
            "gemm->dot",
            "dot->axpy",
        ];
        // One round of one pair on fresh data; true if the second call saw the first one's result.
        let round = |q: &Queue, pair: &str| -> bool {
            let x = Mem::new(q, &[1.0_f64, 2.0, 3.0, 4.0]);
            let y = Mem::new(q, &[1.0_f64, 1.0, 1.0, 1.0]);
            let r = Mem::new(q, &[0.0_f64]);
            let mut m = [0.0_f64; 16];
            for i in 0..4 {
                m[i * 5] = 2.0;
            }
            let (m, ones) = (Mem::new(q, &m), Mem::new(q, &[1.0_f64; 16]));
            let (z, w) = (Mem::new(q, &[0.0_f64; 16]), Mem::new(q, &[0.0_f64; 16]));
            let mut ev: Vec<*mut c_void> = Vec::new();
            let ok = unsafe {
                match pair {
                    "scal->axpy" => {
                        ev.push(bf_dscal_async(q.0, 4, 2.0, x.p, 1, NO_DEPS, 0));
                        ev.push(bf_daxpy_async(q.0, 4, 1.0, x.p, 1, y.p, 1, NO_DEPS, 0));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        y.get(4) == vec![3.0, 5.0, 7.0, 9.0]
                    }
                    "copy->dot" => {
                        ev.push(bf_dcopy_async(q.0, 4, x.p, 1, y.p, 1, NO_DEPS, 0));
                        ev.push(bf_ddot_async(q.0, 4, y.p, 1, y.p, 1, r.p, NO_DEPS, 0));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        r.get(1)[0] == 30.0
                    }
                    "axpy->dot" => {
                        ev.push(bf_daxpy_async(q.0, 4, 2.0, x.p, 1, y.p, 1, NO_DEPS, 0));
                        ev.push(bf_ddot_async(q.0, 4, x.p, 1, y.p, 1, r.p, NO_DEPS, 0));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        r.get(1)[0] == 70.0
                    }
                    "axpy->asum" => {
                        ev.push(bf_daxpy_async(q.0, 4, 1.0, x.p, 1, y.p, 1, NO_DEPS, 0));
                        ev.push(bf_dasum_async(q.0, 4, y.p, 1, r.p, NO_DEPS, 0));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        r.get(1)[0] == 14.0
                    }
                    "axpy->nrm2" => {
                        ev.push(bf_daxpy_async(q.0, 4, 1.0, x.p, 1, y.p, 1, NO_DEPS, 0));
                        ev.push(bf_dnrm2_async(q.0, 4, y.p, 1, r.p, NO_DEPS, 0));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        (r.get(1)[0] - (4.0_f64 + 9.0 + 16.0 + 25.0).sqrt()).abs() < 1e-12
                    }
                    "gemm->gemm" => {
                        ev.push(bf_dgemm_async(
                            q.0, 0, 0, 4, 4, 4, 1.0, m.p, 4, ones.p, 4, 0.0, z.p, 4, NO_DEPS, 0,
                        ));
                        ev.push(bf_dgemm_async(
                            q.0, 0, 0, 4, 4, 4, 1.0, m.p, 4, z.p, 4, 0.0, w.p, 4, NO_DEPS, 0,
                        ));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        w.get(16).iter().all(|e| *e == 4.0)
                    }
                    "gemm->dot" => {
                        ev.push(bf_dgemm_async(
                            q.0, 0, 0, 4, 4, 4, 1.0, m.p, 4, ones.p, 4, 0.0, z.p, 4, NO_DEPS, 0,
                        ));
                        ev.push(bf_ddot_async(q.0, 16, z.p, 1, z.p, 1, r.p, NO_DEPS, 0));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        r.get(1)[0] == 64.0
                    }
                    _ => {
                        ev.push(bf_ddot_async(q.0, 4, x.p, 1, x.p, 1, r.p, NO_DEPS, 0));
                        ev.push(bf_daxpy_async(q.0, 1, 1.0, r.p, 1, y.p, 1, NO_DEPS, 0));
                        assert_eq!(bf_queue_wait(q.0), 0, "{}", last_error());
                        y.get(4)[0] == 31.0
                    }
                }
            };
            assert!(ev.iter().all(|e| !e.is_null()), "{}", last_error());
            free_all(&ev);
            ok
        };
        // Restore the default even if an assertion below unwinds.
        struct ChainOn;
        impl Drop for ChainOn {
            fn drop(&mut self) {
                unsafe { bf_test_set_async_chain(1) };
            }
        }
        let _restore = ChainOn;
        let q = Queue::new();
        let mut off_report = Vec::new();
        let mut on_wrong = Vec::new();
        for pair in PAIRS {
            unsafe { bf_test_set_async_chain(0) };
            let off = (0..N).filter(|_| !round(&q, pair)).count();
            unsafe { bf_test_set_async_chain(1) };
            let on = (0..N).filter(|_| !round(&q, pair)).count();
            off_report.push(format!("{pair}={off}"));
            if on != 0 {
                on_wrong.push(format!("{pair}={on}"));
            }
        }
        eprintln!(
            "ONEMKL-ORDER wrong of {N} with the chain OFF (informational): {}",
            off_report.join(" ")
        );
        assert!(
            on_wrong.is_empty(),
            "with the chain ON, calls read stale data (of {N}): {on_wrong:?}"
        );
    });
}

/// A failing asynchronous task submitted just before a blocking call: the blocking call reports
/// its error (status 1, the message in `bf_last_error`) and the queue's store is empty afterwards,
/// so the error is reported once (ROADMAP R3d). Before the fix `bf_memcpy` waited for the task,
/// returned 0 and left the error for the next `bf_queue_wait` (measured, Windows, oneAPI 2026.1,
/// CPU/OpenCL, 3 of 3 runs). The other entry points end in the same wait; each is checked here.
fn blocking_call_reports_a_failure_behind_it(name: &str, call: impl FnOnce(&Queue) -> i32) {
    let q = Queue::new();
    let e = unsafe { bf_test_async_failure(q.0) };
    assert!(!e.is_null(), "{name}: {}", last_error());
    unsafe { bf_event_free(e) };
    let rc = call(&q);
    assert_eq!(rc, 1, "{name}: expected the asynchronous error, got {rc}");
    assert!(
        last_error().contains("forced asynchronous failure"),
        "{name}: {}",
        last_error()
    );
    // Reported once: nothing is left for the next wait.
    assert_eq!(unsafe { bf_queue_wait(q.0) }, 0, "{name}: {}", last_error());
}

#[test]
fn a_blocking_call_behind_a_failing_task_reports_its_error() {
    watchdog("blocking_call_reports_error", 120, || {
        let _g = serial();
        blocking_call_reports_a_failure_behind_it("bf_memcpy", |q| {
            let src = Mem::new(q, &[1.0_f64; 4]);
            let dst = Mem::new(q, &[0.0_f64; 4]);
            let rc = unsafe { bf_memcpy(q.0, dst.p.cast(), src.p.cast(), 32) };
            assert_eq!(
                dst.get(4),
                vec![1.0; 4],
                "the copy itself must have happened"
            );
            rc
        });
        blocking_call_reports_a_failure_behind_it("bf_saxpy", |q| {
            let x = Mem::new(q, &[1.0_f32; 4]);
            let y = Mem::new(q, &[1.0_f32; 4]);
            unsafe { bf_saxpy(q.0, 4, 2.0, x.p, 1, y.p, 1) }
        });
        blocking_call_reports_a_failure_behind_it("bf_sdot", |q| {
            let x = Mem::new(q, &[1.0_f32; 4]);
            let y = Mem::new(q, &[1.0_f32; 4]);
            let r = Mem::new(q, &[0.0_f32; 1]);
            unsafe { bf_sdot(q.0, 4, x.p, 1, y.p, 1, r.p) }
        });
        blocking_call_reports_a_failure_behind_it("bf_sgemm", |q| {
            let a = Mem::new(q, &[1.0_f32; 4]);
            let b = Mem::new(q, &[1.0_f32; 4]);
            let c = Mem::new(q, &[0.0_f32; 4]);
            unsafe { bf_sgemm(q.0, 0, 0, 2, 2, 2, 1.0, a.p, 2, b.p, 2, 0.0, c.p, 2) }
        });
        blocking_call_reports_a_failure_behind_it("bf_dgemm", |q| {
            let a = Mem::new(q, &[1.0_f64; 4]);
            let b = Mem::new(q, &[1.0_f64; 4]);
            let c = Mem::new(q, &[0.0_f64; 4]);
            unsafe { bf_dgemm(q.0, 0, 0, 2, 2, 2, 1.0, a.p, 2, b.p, 2, 0.0, c.p, 2) }
        });
        blocking_call_reports_a_failure_behind_it("bf_dft_compute", |q| {
            let dims = [4_i64];
            let h = unsafe { bf_dft_c2c_new(q.0, 1, 1, dims.as_ptr(), 1, 1.0, 1.0) };
            assert!(!h.is_null(), "{}", last_error());
            let data = Mem::new(q, &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            let rc = unsafe { bf_dft_compute(h, 0, data.p.cast(), std::ptr::null_mut()) };
            unsafe { bf_dft_free(h) };
            rc
        });
    });
}
