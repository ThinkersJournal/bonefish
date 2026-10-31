//! A SYCL queue.

use std::ffi::{c_char, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

use bonefish_sys as sys;

use crate::error::{Error, Result, check, last_message};

/// Which device to ask the runtime for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// The runtime's default selector.
    Default,
    /// A CPU device (the OpenCL CPU runtime).
    Cpu,
    /// A GPU device.
    Gpu,
}

struct Inner {
    raw: NonNull<c_void>,
    /// Created with profiling, so events from it can report device time.
    profiling: bool,
}

// SAFETY: sharing and moving a queue between threads rests on two different things, and the
// second is NOT the specification:
//
// 1. SYCL calls (queue creation, allocation, copies, free). SYCL 2020 (rev 12), chapter 4
//    introduction, verbatim: "All functions defined in this specification are thread-safe, unless
//    otherwise specified." Section 4.6.5 (queue) reported no exception. Tested: four threads
//    allocating and freeing with shared and per-thread queues passed 25 of 25 runs in a
//    standalone C++ program (oneAPI 2026.1, Windows, CPU device).
// 2. oneMKL calls (BLAS, DFT). The oneAPI specification says, verbatim, "All oneMKL member and
//    non-member functions shall be host thread safe." **The shipped oneMKL did not honour that.**
//    Measured with oneAPI 2026.1 on Windows, CPU/OpenCL device: concurrent first use of
//    `gemm` from four threads crashed inside `mkl_sycl_blas.6.dll` in 28 of 40 runs of a
//    standalone C++ program, and in 15 of 15 fresh-process runs of the Rust threading tests
//    (exception 0xC06D007F or an access violation); on Linux CI (oneAPI 2026.2) the same tests
//    hung. So every oneMKL call in the shim takes one process-wide mutex (`bf::MklLock`), which
//    makes these calls thread-safe by our construction, not by Intel's claim: 15 of 15 and 20 of
//    20 fresh-process runs pass with it. See `ROADMAP.md` R3b for the full numbers and the A/B.
//
// The spec also says "oneMKL objects in class-based APIs may not be shared between concurrent host
// threads unless otherwise specified", and `Dft` stays neither `Send` nor `Sync` until a test of
// sharing one plan says otherwise (the lock serialises the calls, which is not the same as having
// shown that sharing a plan is sound).
//
// Limits: spec texts were read through a summariser and cross-checked by a second search; no
// Intel product-level thread-safety statement was found; the stress results are CPU-device only
// and no GPU has run any of this.
unsafe impl Send for Inner {}
// SAFETY: as above.
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `bf_queue_new` and is freed exactly once, here, after every
        // buffer and plan that holds this queue has been dropped (they hold an `Arc`). The shim
        // does not destroy it at once: it retires the queue and destroys it after a grace
        // period, because an error posted a moment after the last wait would otherwise reach a
        // destroyed queue and terminate the process (ROADMAP R10).
        unsafe { sys::bf_queue_free(self.raw.as_ptr()) }
    }
}

/// A SYCL queue on one device. Cloning shares the same queue; it is released when the last clone
/// and the last buffer or plan made from it are dropped.
///
/// `Send + Sync`: one queue may be shared by several host threads, which may each allocate, copy
/// and call [`blas::gemm`](crate::blas::gemm) on it.
///
/// Read this before relying on it. The oneAPI specification promises that oneMKL calls are host
/// thread safe, and the shipped oneMKL tested here (2026.1, Windows, CPU device) is **not**: its
/// first concurrent use crashes. This crate therefore serialises every oneMKL call behind one
/// process-wide lock, so oneMKL calls from different threads take turns (SYCL allocation and
/// copies still run concurrently). The thread-safety of those calls is the lock's, not Intel's.
/// Details, numbers and the unverified parts (any GPU, other oneMKL releases) are in the source
/// comment above and in `ROADMAP.md` R3b. A [`Dft`](crate::Dft) made from a queue is not
/// shareable between threads.
#[derive(Clone)]
pub struct Queue(Arc<Inner>);

impl Queue {
    /// Create a queue.
    ///
    /// # Errors
    ///
    /// [`Error::Unavailable`] when built without oneAPI; [`Error::Sycl`] when the runtime has no
    /// such device.
    pub fn new(kind: DeviceKind) -> Result<Queue> {
        let code = match kind {
            DeviceKind::Default => 0,
            DeviceKind::Cpu => 1,
            DeviceKind::Gpu => 2,
        };
        // SAFETY: plain call; a null return is handled by `from_raw`.
        Queue::from_raw(unsafe { sys::bf_queue_new(code) }, false)
    }

    /// Create a queue whose events can report device time ([`Event::device_time`](crate::Event::device_time)). Profiling can
    /// only be chosen when the queue is made. It may add overhead to every submission; a
    /// measurement that wants the least should compare against [`Scope::noop`](crate::Scope::noop) on the same queue.
    ///
    /// # Errors
    ///
    /// As [`Queue::new`].
    pub fn new_profiling(kind: DeviceKind) -> Result<Queue> {
        let code = match kind {
            DeviceKind::Default => 0,
            DeviceKind::Cpu => 1,
            DeviceKind::Gpu => 2,
        };
        // SAFETY: plain call; a null return is handled by `from_raw`.
        Queue::from_raw(unsafe { sys::bf_queue_new_ex(code, -1, 1) }, true)
    }

    /// Whether this queue was created with profiling.
    #[must_use]
    pub fn is_profiling(&self) -> bool {
        self.0.profiling
    }

    /// Wrap a queue handle a `bf_queue_new*` call returned, or turn its failure into an error.
    pub(crate) fn from_raw(raw: *mut c_void, profiling: bool) -> Result<Queue> {
        match NonNull::new(raw) {
            Some(raw) => {
                let queue = Queue(Arc::new(Inner { raw, profiling }));
                crate::warmup::once(&queue);
                Ok(queue)
            }
            None if !sys::SYCL_AVAILABLE => Err(Error::Unavailable(last_message())),
            None => Err(Error::Sycl(last_message())),
        }
    }

    /// The device's name as the runtime reports it.
    ///
    /// # Errors
    ///
    /// [`Error::Sycl`] if the runtime cannot answer.
    pub fn device_name(&self) -> Result<String> {
        let mut buf = vec![0 as c_char; 256];
        loop {
            // SAFETY: `buf` is valid for `buf.len()` bytes and the shim NUL-terminates within it.
            let full = unsafe {
                sys::bf_queue_device_name(self.raw(), buf.as_mut_ptr(), buf.len() as i64)
            };
            if full < 0 {
                return Err(Error::Sycl(last_message()));
            }
            if (full as usize) < buf.len() {
                let bytes: Vec<u8> = buf[..full as usize].iter().map(|c| *c as u8).collect();
                return Ok(String::from_utf8_lossy(&bytes).into_owned());
            }
            buf.resize(full as usize + 1, 0);
        }
    }

    /// Whether `other` is the same underlying queue (not merely an equal device).
    #[must_use]
    pub fn same_as(&self, other: &Queue) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn raw(&self) -> *mut c_void {
        self.0.raw.as_ptr()
    }

    /// Turn a shim status into a result. Convenience for the sibling modules.
    pub(crate) fn status(code: i32) -> Result<()> {
        check(code)
    }
}

impl std::fmt::Debug for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Queue").field("ptr", &self.0.raw).finish()
    }
}
