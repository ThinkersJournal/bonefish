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
/// reads what the previous wrote, so a wrong order shows as a wrong power of two. This is the
/// property the scoped API's soundness rests on (design note 3.4).
///
/// **This test has no teeth on the CPU/OpenCL device.** With the `in_order` property removed from
/// the shim, an out-of-order queue still passed it 15 of 15 runs, so it cannot tell the two apart
/// here; `queues_are_in_order` (the property itself) is what actually pins the requirement. It
/// stays as the end-to-end check that would fail on a device that really reorders. GPU UNVERIFIED.
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
