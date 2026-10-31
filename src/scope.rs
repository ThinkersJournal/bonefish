//! Non-blocking operations inside a scope (R3c step 2; design: `docs/ASYNC-DESIGN.md`).
//!
//! ```text
//! queue.scope(|s| {
//!     let a = s.read(&a_buf)?;       // Read<'scope, T>:  holds &'scope UsmBuffer
//!     let x = s.write(&mut x_buf)?;  // Write<'scope, T>: holds &'scope mut UsmBuffer
//!     let y = s.write(&mut y_buf)?;
//!     let e = s.gemm(Transpose::No, Transpose::No, n, n, n, 1.0, &a, n, &x, n, 0.0, &y, n)?;
//!     s.gemm_after(&[&e], /* next product, reading y */)?;
//!     Ok(())
//! })?;                               // waits for everything, then returns the first async error
//! ```
//!
//! # Why this is sound
//!
//! Soundness does not depend on any handle's `Drop` (`mem::forget` on an [`Event`] would defeat
//! that). It rests on four things:
//!
//! 1. **The host cannot touch a bound buffer.** [`Scope::read`] keeps a shared borrow and
//!    [`Scope::write`] an exclusive one, both for the whole scope, so `to_vec`, `as_slice`,
//!    `copy_from_slice` or dropping the buffer are rejected by the borrow checker (the
//!    `compile_fail` examples on [`Queue::scope`] pin that).
//! 2. **The scope waits on every way out.** A guard waits for the queue on normal return, on
//!    `?` and on panic unwinding. The closure runs inside [`Queue::scope`], so nothing outside
//!    can forget the guard, and `'scope` is invariant so no handle can be smuggled out.
//! 3. **Queues are in-order**, so two operations that name the same [`Write`] cannot run at the
//!    same time on the device. (The CPU/OpenCL device also happens to run an out-of-order queue
//!    in order, so no test here can tell; this rests on the SYCL specification and on the
//!    `in_order` property the shim sets. GPU UNVERIFIED.)
//! 4. **One operation never aliases its own output**: an input that is the same buffer as the
//!    output is an [`Error::InvalidArgument`], because BLAS forbids it.
//!
//! A [`Scope`] is `!Sync`: one scope, one thread. Several threads each use their own scope (and
//! the shim's oneMKL lock serialises the submissions, see `ROADMAP.md` R3b).

use std::cell::Cell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use bonefish_sys as sys;

use crate::blas::{Transpose, check_matrix, dim};
use crate::error::{Error, Result, check, last_message};
use crate::queue::Queue;
use crate::scalar::{Element, Scalar};
use crate::usm::UsmBuffer;

mod sealed {
    pub trait Sealed {}
}

/// A buffer bound for reading inside a scope. Holds a shared borrow of the buffer until the
/// scope ends.
pub struct Read<'scope, T: Element> {
    buf: &'scope UsmBuffer<T>,
}

/// A buffer bound for writing inside a scope. Holds an exclusive borrow of the buffer until the
/// scope ends. Operations take it by `&`, so several operations can name it; the in-order queue
/// keeps them from overlapping.
pub struct Write<'scope, T: Element> {
    buf: &'scope mut UsmBuffer<T>,
}

/// Something an operation can read: a [`Read`] or a [`Write`] (a result of an earlier operation).
/// Sealed.
pub trait Operand<T: Element>: sealed::Sealed {
    #[doc(hidden)]
    fn buffer(&self) -> &UsmBuffer<T>;
}

impl<T: Element> sealed::Sealed for Read<'_, T> {}
impl<T: Element> sealed::Sealed for Write<'_, T> {}
impl<T: Element> Operand<T> for Read<'_, T> {
    fn buffer(&self) -> &UsmBuffer<T> {
        self.buf
    }
}
impl<T: Element> Operand<T> for Write<'_, T> {
    fn buffer(&self) -> &UsmBuffer<T> {
        self.buf
    }
}

/// A handle to submitted device work. It cannot outlive its scope. Dropping it only releases the
/// handle (the work still completes and the scope still waits for it), so leaking one costs
/// memory, never safety.
pub struct Event<'scope> {
    raw: NonNull<c_void>,
    _scope: PhantomData<&'scope mut &'scope ()>,
}

impl Event<'_> {
    /// Wait for this operation and everything before it on the queue.
    ///
    /// # Errors
    ///
    /// The asynchronous error, if the operation failed after submission.
    pub fn wait(&self) -> Result<()> {
        // SAFETY: `raw` is a live event handle owned by `self`.
        check(unsafe { sys::bf_event_wait(self.raw.as_ptr()) })
    }
}

impl Drop for Event<'_> {
    fn drop(&mut self) {
        // SAFETY: `raw` came from a `*_async` call and is freed exactly once, here. Freeing a
        // handle does not cancel or wait for the work.
        unsafe { sys::bf_event_free(self.raw.as_ptr()) }
    }
}

impl std::fmt::Debug for Event<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Event").finish_non_exhaustive()
    }
}

/// Where non-blocking operations are submitted. Made by [`Queue::scope`].
///
/// `Send` but not `Sync`: one scope, one thread. The control first (it is `Send`), then the
/// claim:
///
/// ```
/// fn needs_send<T: Send>() {}
/// needs_send::<bonefish::Scope<'static, 'static>>();
/// ```
///
/// ```compile_fail,E0277
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<bonefish::Scope<'static, 'static>>();
/// ```
pub struct Scope<'scope, 'env: 'scope> {
    queue: &'env Queue,
    // Invariant in both lifetimes, as in `std::thread::scope`: no handle can leave the scope.
    _scope: PhantomData<&'scope mut &'scope ()>,
    _env: PhantomData<&'env mut &'env ()>,
    // `!Sync`: one scope, one thread.
    _not_sync: PhantomData<Cell<()>>,
}

/// Waits for the queue on drop unless disarmed. Covers panic unwinding out of the closure.
struct WaitOnDrop<'a> {
    queue: &'a Queue,
    armed: bool,
}

impl Drop for WaitOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            // SAFETY: the queue is alive (borrowed). The result is ignored: this runs while
            // unwinding, where there is nowhere to report it; the point is only that the device
            // is done with the operands before their borrows end.
            unsafe { sys::bf_queue_wait(self.queue.raw()) };
        }
    }
}

impl Queue {
    /// Run `f` with a [`Scope`] on which operations are submitted without waiting. When `f`
    /// returns (or fails, or panics) the queue is waited on before this function returns, so no
    /// device work outlives the borrows the scope took.
    ///
    /// Returns `f`'s error if it failed, otherwise the first asynchronous error from the work,
    /// otherwise `f`'s value.
    ///
    /// # Errors
    ///
    /// Whatever `f` returns, or an asynchronous error from the submitted work.
    ///
    /// # Examples
    ///
    /// The host cannot read a buffer that is bound for writing, nor drop one that is bound for
    /// reading, while the scope runs. First the control: this compiles (it is not run here, a
    /// doctest has no device).
    ///
    /// ```no_run
    /// use bonefish::{DeviceKind, Queue, Transpose, UsmBuffer, UsmKind};
    /// # fn main() -> bonefish::Result<()> {
    /// let q = Queue::new(DeviceKind::Cpu)?;
    /// let a = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let b = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let mut c = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// q.scope(|s| {
    ///     let (a, b) = (s.read(&a)?, s.read(&b)?);
    ///     let c = s.write(&mut c)?;
    ///     s.gemm(Transpose::No, Transpose::No, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &c, 2)?;
    ///     Ok(())
    /// })?;
    /// let _ = c.to_vec()?; // fine: the scope has ended
    /// # Ok(()) }
    /// ```
    ///
    /// Reading `c` on the host while it is bound for writing does not compile:
    ///
    /// ```compile_fail,E0502
    /// use bonefish::{DeviceKind, Queue, Transpose, UsmBuffer, UsmKind};
    /// # fn main() -> bonefish::Result<()> {
    /// let q = Queue::new(DeviceKind::Cpu)?;
    /// let a = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let b = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let mut c = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// q.scope(|s| {
    ///     let (a, b) = (s.read(&a)?, s.read(&b)?);
    ///     let cw = s.write(&mut c)?;
    ///     s.gemm(Transpose::No, Transpose::No, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &cw, 2)?;
    ///     let _ = c.to_vec()?; // the device may still be writing c
    ///     Ok(())
    /// })?;
    /// # Ok(()) }
    /// ```
    ///
    /// Dropping `a` while it is bound for reading does not compile:
    ///
    /// ```compile_fail,E0505
    /// use bonefish::{DeviceKind, Queue, Transpose, UsmBuffer, UsmKind};
    /// # fn main() -> bonefish::Result<()> {
    /// let q = Queue::new(DeviceKind::Cpu)?;
    /// let a = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let b = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let mut c = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// q.scope(|s| {
    ///     let (ar, br) = (s.read(&a)?, s.read(&b)?);
    ///     let c = s.write(&mut c)?;
    ///     s.gemm(Transpose::No, Transpose::No, 2, 2, 2, 1.0, &ar, 2, &br, 2, 0.0, &c, 2)?;
    ///     drop(a); // the device may still be reading a
    ///     Ok(())
    /// })?;
    /// # Ok(()) }
    /// ```
    ///
    /// An event cannot be carried out of the scope:
    ///
    /// ```compile_fail
    /// use bonefish::{DeviceKind, Queue, Transpose, UsmBuffer, UsmKind};
    /// # fn main() -> bonefish::Result<()> {
    /// let q = Queue::new(DeviceKind::Cpu)?;
    /// let a = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let b = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let mut c = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4)?;
    /// let escaped = q.scope(|s| {
    ///     let (a, b) = (s.read(&a)?, s.read(&b)?);
    ///     let c = s.write(&mut c)?;
    ///     s.gemm(Transpose::No, Transpose::No, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &c, 2)
    /// })?;
    /// drop(escaped);
    /// # Ok(()) }
    /// ```
    pub fn scope<'env, F, R>(&'env self, f: F) -> Result<R>
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> Result<R>,
    {
        let scope = Scope {
            queue: self,
            _scope: PhantomData,
            _env: PhantomData,
            _not_sync: PhantomData,
        };
        let mut guard = WaitOnDrop {
            queue: self,
            armed: true,
        };
        let result = f(&scope);
        // Normal path (including `Err` from the closure): wait here so the error can be reported.
        guard.armed = false;
        // SAFETY: the queue is alive (borrowed).
        let waited = check(unsafe { sys::bf_queue_wait(self.raw()) });
        match (result, waited) {
            (Err(e), _) | (Ok(_), Err(e)) => Err(e),
            (Ok(v), Ok(())) => Ok(v),
        }
    }
}

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// Bind `buf` for reading until the scope ends.
    ///
    /// # Errors
    ///
    /// [`Error::QueueMismatch`] if the buffer belongs to another queue.
    pub fn read<T: Element>(&'scope self, buf: &'scope UsmBuffer<T>) -> Result<Read<'scope, T>> {
        self.own(buf)?;
        Ok(Read { buf })
    }

    /// Bind `buf` for writing (and for use as an input of later operations) until the scope ends.
    ///
    /// # Errors
    ///
    /// [`Error::QueueMismatch`] if the buffer belongs to another queue.
    pub fn write<T: Element>(
        &'scope self,
        buf: &'scope mut UsmBuffer<T>,
    ) -> Result<Write<'scope, T>> {
        self.own(buf)?;
        Ok(Write { buf })
    }

    fn own<T: Element>(&self, buf: &UsmBuffer<T>) -> Result<()> {
        if self.queue.same_as(buf.queue()) {
            Ok(())
        } else {
            Err(Error::QueueMismatch)
        }
    }

    /// Non-blocking `c = alpha * op(a) * op(b) + beta * c`, column-major. Returns once the work is
    /// submitted; the device finishes it later, at the latest when the scope ends.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an extent that does not fit its buffer, or when `c` is the
    /// same buffer as `a` or `b`; [`Error::Sycl`]/[`Error::Mkl`]/[`Error::Other`] if submission
    /// fails.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<T: Scalar>(
        &'scope self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        b: &impl Operand<T>,
        ldb: usize,
        beta: T,
        c: &Write<'scope, T>,
        ldc: usize,
    ) -> Result<Event<'scope>> {
        self.gemm_after(
            &[],
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
    }

    /// As [`Scope::gemm`], but the work also waits for every event in `after`. On an in-order
    /// queue earlier work is already ordered; this is for stating a dependency explicitly.
    ///
    /// # Errors
    ///
    /// As [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_after<T: Scalar>(
        &'scope self,
        after: &[&Event<'scope>],
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        b: &impl Operand<T>,
        ldb: usize,
        beta: T,
        c: &Write<'scope, T>,
        ldc: usize,
    ) -> Result<Event<'scope>> {
        let (a, b, c) = (a.buffer(), b.buffer(), &*c.buf);
        let (a_rows, a_cols) = if transa == Transpose::No {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if transb == Transpose::No {
            (k, n)
        } else {
            (n, k)
        };
        check_matrix("A", a.len(), lda, a_rows, a_cols)?;
        check_matrix("B", b.len(), ldb, b_rows, b_cols)?;
        check_matrix("C", c.len(), ldc, m, n)?;
        if std::ptr::eq(a.raw(), c.raw()) || std::ptr::eq(b.raw(), c.raw()) {
            return Err(Error::InvalidArgument(
                "the output buffer is also an input of the same operation".into(),
            ));
        }
        let deps: Vec<*mut c_void> = after.iter().map(|e| e.raw.as_ptr()).collect();
        // SAFETY: every pointer is USM from this scope's queue (checked when bound) and the
        // extents above fit inside the buffers. The buffers stay borrowed until the scope ends,
        // and the scope waits for the queue before that, so the device is finished with them
        // first. `c` is distinct from `a` and `b` (checked above), and the in-order queue keeps
        // this operation from overlapping earlier ones that name the same buffers. `deps` are
        // live events of this scope.
        let raw = unsafe {
            T::gemm_async(
                self.queue.raw(),
                transa.code(),
                transb.code(),
                dim("m", m)?,
                dim("n", n)?,
                dim("k", k)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                b.raw(),
                dim("ldb", ldb)?,
                beta,
                c.raw(),
                dim("ldc", ldc)?,
                if deps.is_empty() {
                    std::ptr::null()
                } else {
                    deps.as_ptr()
                },
                deps.len() as i64,
            )
        };
        match NonNull::new(raw) {
            Some(raw) => Ok(Event {
                raw,
                _scope: PhantomData,
            }),
            None if !sys::SYCL_AVAILABLE => Err(Error::Unavailable(last_message())),
            None => Err(Error::Other(format!(
                "submission failed: {}",
                last_message()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceKind, UsmKind};

    /// A scope whose closure returns while the queue holds a throwing host task must report that
    /// asynchronous error from `scope`, not abort and not swallow it. Needs a device.
    #[test]
    fn an_asynchronous_error_is_returned_by_scope() {
        if !crate::SYCL_AVAILABLE {
            eprintln!("SKIP: built without oneAPI. Nothing ran on a device.");
            return;
        }
        let q = Queue::new(DeviceKind::Cpu).expect("a CPU SYCL device");
        // Control: an empty scope is Ok.
        assert!(q.scope(|_| Ok(1)).is_ok());
        let r: Result<()> = q.scope(|_s| {
            // SAFETY: the queue handle is live; the event is freed below.
            let e = unsafe { sys::bf_test_async_failure(q.raw()) };
            assert!(!e.is_null());
            unsafe { sys::bf_event_free(e) };
            Ok(())
        });
        match r {
            Err(Error::Sycl(m)) => assert!(m.contains("forced asynchronous failure"), "{m}"),
            other => panic!("expected the asynchronous SYCL error, got {other:?}"),
        }
        // And it is not reported again.
        assert!(q.scope(|_| Ok(())).is_ok());
    }

    /// The closure's own error wins over (and does not hide) a clean wait.
    #[test]
    fn a_closure_error_is_returned_after_the_wait() {
        if !crate::SYCL_AVAILABLE {
            return;
        }
        let q = Queue::new(DeviceKind::Cpu).expect("a CPU SYCL device");
        let buf = UsmBuffer::<f32>::zeroed(&q, UsmKind::Shared, 4).unwrap();
        let r: Result<()> = q.scope(|s| {
            let _r = s.read(&buf)?;
            Err(Error::Other("closure failed".into()))
        });
        assert_eq!(r, Err(Error::Other("closure failed".into())));
    }
}
