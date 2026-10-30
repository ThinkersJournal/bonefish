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
}

// SAFETY: sharing and moving a queue between threads is sound under the two specification
// statements this crate relies on, both quoted verbatim:
//
// - SYCL 2020 (rev 12), chapter 4 introduction: "All functions defined in this specification are
//   thread-safe, unless otherwise specified." Section 4.6.5 (queue) reported no exception.
// - oneAPI specification (oneMKL, "Execution Model > Host Thread Safety", v1.3-rev-1): "All oneMKL
//   member and non-member functions shall be host thread safe." That covers the non-member BLAS
//   calls this crate makes on a queue. The same section adds: "oneMKL objects in class-based APIs
//   may not be shared between concurrent host threads unless otherwise specified." which is why
//   `Dft` is NOT `Sync` (nor `Send`, conservatively).
//
// Limits, stated rather than hidden: these are specification texts, read through a summariser
// and cross-checked by a second search; Intel's own developer-reference pages for the shipped
// product were searched and no product-level statement was found. `ROADMAP.md` (R3b) has the
// detail, and `tests/threads.rs` exercises concurrent use without proving it.
unsafe impl Send for Inner {}
// SAFETY: as above.
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `bf_queue_new` and is freed exactly once, here, after every
        // buffer and plan that holds this queue has been dropped (they hold an `Arc`).
        unsafe { sys::bf_queue_free(self.raw.as_ptr()) }
    }
}

/// A SYCL queue on one device. Cloning shares the same queue; it is released when the last clone
/// and the last buffer or plan made from it are dropped.
///
/// `Send + Sync`: one queue may be shared by several host threads, which may each allocate, copy
/// and call [`blas::gemm`](crate::blas::gemm) on it. That rests on the SYCL 2020 and oneAPI
/// (oneMKL) **specification** statements quoted at the `unsafe impl` in the source. Whether the
/// shipped oneMKL (2026.1 was the one used here) conforms to the specification on this point is
/// **unverified**: no product-level statement was found, and the multi-threaded tests are
/// evidence, not proof. See `ROADMAP.md` R3b. A [`Dft`](crate::Dft) made from a queue is not
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
        // SAFETY: plain call; a null return is handled below.
        let raw = unsafe { sys::bf_queue_new(code) };
        match NonNull::new(raw) {
            Some(raw) => Ok(Queue(Arc::new(Inner { raw }))),
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
