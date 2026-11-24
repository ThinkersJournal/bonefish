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
//! 3. **Asynchronous submissions on a queue are chained**, so two operations that name the same
//!    [`Write`] cannot run at the same time on the device: the shim makes every asynchronous call
//!    depend on the queue's previous one. This is deliberately NOT left to the queue being
//!    in-order. Measured (Windows, oneAPI 2026.1, CPU/OpenCL, in-order queue): a oneMKL `dot`
//!    submitted right after an `axpy`, `copy` or `gemm` read the old data in 99 to 100 of 100
//!    rounds; with the chain it never did. The chain can be switched off for diagnosis with
//!    `BONEFISH_UNSAFE_NO_ASYNC_CHAIN=1`; do not do that in a program that matters. Other
//!    oneMKL releases and any GPU are UNVERIFIED (`ROADMAP.md` R3c).
//! 4. **One operation never aliases its own output**: an input that is the same buffer as the
//!    output is an [`Error::InvalidArgument`], because BLAS forbids it.
//!
//! Operations: [`Scope::gemm`], level 1 ([`Scope::axpy`], [`Scope::scal`], [`Scope::copy`],
//! [`Scope::swap`], [`Scope::dot`], [`Scope::nrm2`], [`Scope::asum`], [`Scope::iamax`]),
//! [`Scope::gemv`] and DFT ([`Scope::plan`], [`Scope::dft_forward`] and friends).
//!
//! A [`Scope`] is `!Sync`: one scope, one thread. Several threads each use their own scope (and
//! the shim's oneMKL lock serialises the submissions, see `ROADMAP.md` R3b).

use std::cell::Cell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use bonefish_sys as sys;

use crate::blas::{Transpose, check_matrix, dim};
use crate::dft::{Dft, Placement};
use crate::error::{Error, Result, check, last_message};
use crate::queue::Queue;
use crate::rdft::RealDft;
use crate::scalar::{BlasElement, Complex, Element, Scalar};
use crate::usm::UsmBuffer;

mod batched;
mod dft;
mod event;
mod gemm;
mod half_batched;
mod half_gemm;
mod level1;
mod level2;
mod level3;

mod sealed {
    pub trait Sealed {}
}

/// A buffer bound for reading inside a scope. Holds a shared borrow of the buffer until the
/// scope ends.
pub struct Read<'scope, T: Element> {
    buf: &'scope UsmBuffer<T>,
}

/// A buffer bound for writing inside a scope. Holds an exclusive borrow of the buffer until the
/// scope ends. Operations take it by `&`, so several operations can name it; the shim's chain of
/// asynchronous submissions keeps them from overlapping.
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
    /// Returns `f`'s error if it failed, otherwise the first asynchronous error the queue has
    /// recorded, otherwise `f`'s value.
    ///
    /// # Errors
    ///
    /// Whatever `f` returns, or an asynchronous error from the submitted work.
    ///
    /// # Asynchronous errors can arrive late
    ///
    /// The memory guarantee above does not depend on this, but error reporting does: `Ok` is
    /// **not proof** that the work succeeded. Errors are recorded per queue and reported by the
    /// first wait that finds them. Measured on Windows (oneAPI 2026.1, CPU device, a failing host
    /// task forced as the error): with nothing else running, 0 of 150 errors were missed by the
    /// scope that submitted the work; with four threads each failing on their own queue, about
    /// 2 to 5 in 100 were not yet posted when that wait returned, and every one of them was
    /// reported by a later wait on the same queue (66 of 66). So an error can surface from a later
    /// scope on the same queue. It was never reported on another queue (0 of 9 runs, after a fix:
    /// an earlier version kept errors per thread and lost about half of them). A failing *kernel*
    /// was not measured; the forced error is a throwing host task. GPU UNVERIFIED.
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

    /// Wrap the event handle a `*_async` call returned, or turn its failure into an error.
    fn event(&'scope self, raw: *mut c_void) -> Result<Event<'scope>> {
        match NonNull::new(raw) {
            Some(raw) => Ok(Event::new(raw, self.queue.is_profiling())),
            None if !sys::SYCL_AVAILABLE => Err(Error::Unavailable(last_message())),
            None => Err(Error::Other(format!(
                "submission failed: {}",
                last_message()
            ))),
        }
    }
}

/// `n` elements at stride `inc` must fit in a buffer of `len` elements.
fn check_vec(name: &str, len: usize, n: usize, inc: usize) -> Result<()> {
    if inc == 0 {
        return Err(Error::InvalidArgument(format!(
            "{name}: the increment must be at least 1"
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let needed = (n - 1)
        .checked_mul(inc)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| Error::InvalidArgument(format!("{name}: extent overflows")))?;
    if needed > len {
        return Err(Error::InvalidArgument(format!(
            "{name}: {n} elements at stride {inc} need {needed} elements, buffer has {len}"
        )));
    }
    Ok(())
}

/// A scalar result is written to element 0 of its buffer.
fn check_result(name: &str, len: usize) -> Result<()> {
    if len == 0 {
        return Err(Error::InvalidArgument(format!(
            "{name}: the buffer is empty"
        )));
    }
    Ok(())
}

/// The two buffers must be different allocations: an operation never reads and writes the same
/// memory, and BLAS forbids it.
fn distinct<A: Element, B: Element>(
    a: &UsmBuffer<A>,
    a_name: &str,
    b: &UsmBuffer<B>,
    b_name: &str,
) -> Result<()> {
    if std::ptr::eq(a.raw().cast::<c_void>(), b.raw().cast::<c_void>()) {
        return Err(Error::InvalidArgument(format!(
            "{b_name} is the same buffer as {a_name}, which this operation also writes or reads"
        )));
    }
    Ok(())
}

pub use batched::BatchedGemm;
pub use dft::{Plan, RealPlan};
pub use event::{DeviceTime, Event};

const NO_DEPS: *const *mut c_void = std::ptr::null();

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// Submit an empty kernel. Its [`Event::device_time`] on a profiling queue is the launch
    /// floor to compare a real operation's time against. Chained to the previous operation like
    /// any other.
    ///
    /// # Errors
    ///
    /// Submission errors as for [`Scope::gemm`].
    pub fn noop(&'scope self) -> Result<Event<'scope>> {
        // SAFETY: the queue is alive; no pointers other than the null dependency list are passed.
        let raw = unsafe { sys::bf_noop_async(self.queue.raw(), NO_DEPS, 0) };
        self.event(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceKind, UsmKind};

    /// A scope whose closure returns while the queue holds a failing host task must report that
    /// asynchronous error, not abort and not swallow it. Needs a device.
    ///
    /// The error may come from the scope that submitted the work or, rarely, from a later wait on
    /// the same queue: the runtime can complete the wait slightly before it posts the error
    /// (measured, `bonefish-sys/tests/async_shim.rs`). So the test polls further empty scopes for
    /// up to two seconds, and what it requires is that the error arrives, once.
    #[test]
    fn an_asynchronous_error_is_returned_by_scope() {
        if !crate::SYCL_AVAILABLE {
            eprintln!("SKIP: built without oneAPI. Nothing ran on a device.");
            return;
        }
        let q = Queue::new(DeviceKind::Cpu).expect("a CPU SYCL device");
        // Control: an empty scope is Ok.
        assert!(q.scope(|_| Ok(1)).is_ok());
        let mut result: Result<()> = q.scope(|_s| {
            // SAFETY: the queue handle is live; the event is freed below.
            let e = unsafe { sys::bf_test_async_failure(q.raw()) };
            assert!(!e.is_null());
            unsafe { sys::bf_event_free(e) };
            Ok(())
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while result.is_ok() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
            result = q.scope(|_| Ok(()));
        }
        match result {
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
