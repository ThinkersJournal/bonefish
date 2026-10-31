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
use crate::scalar::{Complex, Element, GemmElement, Scalar};
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

/// A handle to submitted device work. It cannot outlive its scope. Dropping it only releases the
/// handle (the work still completes and the scope still waits for it), so leaking one costs
/// memory, never safety.
pub struct Event<'scope> {
    raw: NonNull<c_void>,
    /// Its queue was created with profiling.
    profiling: bool,
    _scope: PhantomData<&'scope mut &'scope ()>,
}

/// Device-side timing of the work behind an [`Event`], in nanoseconds on the device's own clock.
/// Only differences between values from the same device are meaningful.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DeviceTime {
    /// When the work started on the device.
    pub start_ns: u64,
    /// When it finished.
    pub end_ns: u64,
}

impl DeviceTime {
    /// `end - start`.
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

impl Event<'_> {
    /// Device start and end time of this operation (SYCL `command_start` / `command_end`). Waits
    /// for the operation first.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the queue was not created with profiling
    /// ([`Queue::new_profiling`]): an error, never a zero. Otherwise the asynchronous error, if the
    /// operation failed.
    pub fn device_time(&self) -> Result<DeviceTime> {
        if !self.profiling {
            return Err(Error::InvalidArgument(
                "this event's queue was not created with profiling (Queue::new_profiling)".into(),
            ));
        }
        let (mut start_ns, mut end_ns) = (0_u64, 0_u64);
        // SAFETY: `raw` is a live event handle; both outputs are valid for writes.
        check(unsafe {
            sys::bf_event_profiling(self.raw.as_ptr(), &raw mut start_ns, &raw mut end_ns)
        })?;
        Ok(DeviceTime { start_ns, end_ns })
    }

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

    /// Non-blocking `c = alpha * op(a) * op(b) + beta * c`, column-major. Returns once the work is
    /// submitted; the device finishes it later, at the latest when the scope ends.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an extent that does not fit its buffer, or when `c` is the
    /// same buffer as `a` or `b`; [`Error::Sycl`]/[`Error::Mkl`]/[`Error::Other`] if submission
    /// fails.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<T: GemmElement>(
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

    /// As [`Scope::gemm`], but the work also waits for every event in `after`. Earlier work in
    /// the scope is already ordered before this (the shim chains submissions); this is for stating
    /// a dependency explicitly.
    ///
    /// # Errors
    ///
    /// As [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_after<T: GemmElement>(
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
        // first. `c` is distinct from `a` and `b` (checked above), and the shim chains this
        // submission to the queue's previous one, so it cannot overlap earlier ones that name the
        // same buffers. `deps` are live events of this scope.
        let raw = unsafe {
            T::gemm_raw_async(
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
        self.event(raw)
    }

    /// Wrap the event handle a `*_async` call returned, or turn its failure into an error.
    fn event(&'scope self, raw: *mut c_void) -> Result<Event<'scope>> {
        match NonNull::new(raw) {
            Some(raw) => Ok(Event {
                raw,
                profiling: self.queue.is_profiling(),
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

/// A DFT plan bound for use inside a scope. Holds a shared borrow of the plan until the scope
/// ends, so the plan cannot be dropped while a transform that uses it is still running.
pub struct Plan<'scope, T: Scalar> {
    dft: &'scope Dft<T>,
}

const NO_DEPS: *const *mut c_void = std::ptr::null();

/// Level 1, `gemv` and DFT. Each call returns once the work is submitted. Operations on one queue
/// run in submission order because the shim chains each asynchronous submission to the queue's
/// previous one; an in-order queue alone does not order oneMKL calls (`ROADMAP.md` R3c). Vector
/// increments must be at least 1.
impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    /// `y = alpha * x + y`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit its buffer or `y` is the same buffer as
    /// `x`; submission errors as for [`Scope::gemm`].
    pub fn axpy<T: Scalar>(
        &'scope self,
        n: usize,
        alpha: T,
        x: &impl Operand<T>,
        incx: usize,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (x, y) = (x.buffer(), &*y.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        distinct(y, "y", x, "x")?;
        // SAFETY: extents checked; buffers are USM of this scope's queue, borrowed until the scope
        // ends, which waits first; `y` is distinct from `x`.
        let raw = unsafe {
            T::axpy_async(
                self.queue.raw(),
                dim("n", n)?,
                alpha,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `x = alpha * x`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the extent does not fit; submission errors as for
    /// [`Scope::gemm`].
    pub fn scal<T: Scalar>(
        &'scope self,
        n: usize,
        alpha: T,
        x: &Write<'scope, T>,
        incx: usize,
    ) -> Result<Event<'scope>> {
        let x = &*x.buf;
        check_vec("x", x.len(), n, incx)?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::scal_async(
                self.queue.raw(),
                dim("n", n)?,
                alpha,
                x.raw(),
                dim("incx", incx)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `y = x`.
    ///
    /// # Errors
    ///
    /// As [`Scope::axpy`].
    pub fn copy<T: Scalar>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (x, y) = (x.buffer(), &*y.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        distinct(y, "y", x, "x")?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::copy_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// Exchange `x` and `y`.
    ///
    /// # Errors
    ///
    /// As [`Scope::axpy`].
    pub fn swap<T: Scalar>(
        &'scope self,
        n: usize,
        x: &Write<'scope, T>,
        incx: usize,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (x, y) = (&*x.buf, &*y.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        distinct(x, "x", y, "y")?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::swap_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0] = x . y`. The result is written when the work runs, so read it after the scope
    /// (or after waiting for the returned event).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `result` is the same buffer as `x`
    /// or `y`; submission errors as for [`Scope::gemm`].
    pub fn dot<T: Scalar>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        y: &impl Operand<T>,
        incy: usize,
        result: &Write<'scope, T>,
    ) -> Result<Event<'scope>> {
        let (x, y, r) = (x.buffer(), y.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_vec("y", y.len(), n, incy)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        distinct(r, "result", y, "y")?;
        // SAFETY: as `axpy`; `result` has at least one element and is distinct from the inputs.
        let raw = unsafe {
            T::dot_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                y.raw(),
                dim("incy", incy)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0] = ||x||_2`.
    ///
    /// # Errors
    ///
    /// As [`Scope::dot`].
    pub fn nrm2<T: Scalar>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        result: &Write<'scope, T>,
    ) -> Result<Event<'scope>> {
        let (x, r) = (x.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        // SAFETY: as `dot`.
        let raw = unsafe {
            T::nrm2_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0] = sum |x_i|`.
    ///
    /// # Errors
    ///
    /// As [`Scope::dot`].
    pub fn asum<T: Scalar>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        result: &Write<'scope, T>,
    ) -> Result<Event<'scope>> {
        let (x, r) = (x.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        // SAFETY: as `dot`.
        let raw = unsafe {
            T::asum_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `result[0]` = zero-based index of the first element of largest magnitude.
    ///
    /// # Errors
    ///
    /// As [`Scope::dot`].
    pub fn iamax<T: Scalar>(
        &'scope self,
        n: usize,
        x: &impl Operand<T>,
        incx: usize,
        result: &Write<'scope, i64>,
    ) -> Result<Event<'scope>> {
        let (x, r) = (x.buffer(), &*result.buf);
        check_vec("x", x.len(), n, incx)?;
        check_result("result", r.len())?;
        distinct(r, "result", x, "x")?;
        // SAFETY: as `dot`.
        let raw = unsafe {
            T::iamax_async(
                self.queue.raw(),
                dim("n", n)?,
                x.raw(),
                dim("incx", incx)?,
                r.raw(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// `y = alpha * op(A) * x + beta * y`, `A` is `m x n` column-major.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if an extent does not fit or `y` is the same buffer as `a` or
    /// `x`; submission errors as for [`Scope::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemv<T: Scalar>(
        &'scope self,
        trans: Transpose,
        m: usize,
        n: usize,
        alpha: T,
        a: &impl Operand<T>,
        lda: usize,
        x: &impl Operand<T>,
        incx: usize,
        beta: T,
        y: &Write<'scope, T>,
        incy: usize,
    ) -> Result<Event<'scope>> {
        let (a, x, y) = (a.buffer(), x.buffer(), &*y.buf);
        let (x_len, y_len) = if trans == Transpose::No {
            (n, m)
        } else {
            (m, n)
        };
        check_matrix("A", a.len(), lda, m, n)?;
        check_vec("x", x.len(), x_len, incx)?;
        check_vec("y", y.len(), y_len, incy)?;
        distinct(y, "y", a, "A")?;
        distinct(y, "y", x, "x")?;
        // SAFETY: as `axpy`.
        let raw = unsafe {
            T::gemv_async(
                self.queue.raw(),
                trans.code(),
                dim("m", m)?,
                dim("n", n)?,
                alpha,
                a.raw(),
                dim("lda", lda)?,
                x.raw(),
                dim("incx", incx)?,
                beta,
                y.raw(),
                dim("incy", incy)?,
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

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

    /// Bind a DFT plan for use in this scope. While it is bound the plan cannot be dropped, since
    /// the transform that uses it may still be running. First the control, which compiles:
    ///
    /// ```no_run
    /// use bonefish::{Complex, DeviceKind, Dft, Placement, Queue, UsmBuffer, UsmKind};
    /// # fn main() -> bonefish::Result<()> {
    /// let q = Queue::new(DeviceKind::Cpu)?;
    /// let owner = Dft::<f32>::c2c(&q, &[4], Placement::InPlace, 1.0, 1.0)?;
    /// let mut data = UsmBuffer::<Complex<f32>>::zeroed(&q, UsmKind::Shared, 4)?;
    /// q.scope(|s| {
    ///     let plan = s.plan(&owner)?;
    ///     let d = s.write(&mut data)?;
    ///     s.dft_forward_in_place(&plan, &d)?;
    ///     Ok(())
    /// })?;
    /// drop(owner); // fine: the scope has ended
    /// # Ok(()) }
    /// ```
    ///
    /// Dropping the plan inside the scope does not compile:
    ///
    /// ```compile_fail,E0505
    /// use bonefish::{Complex, DeviceKind, Dft, Placement, Queue, UsmBuffer, UsmKind};
    /// # fn main() -> bonefish::Result<()> {
    /// let q = Queue::new(DeviceKind::Cpu)?;
    /// let owner = Dft::<f32>::c2c(&q, &[4], Placement::InPlace, 1.0, 1.0)?;
    /// let mut data = UsmBuffer::<Complex<f32>>::zeroed(&q, UsmKind::Shared, 4)?;
    /// q.scope(|s| {
    ///     let plan = s.plan(&owner)?;
    ///     let d = s.write(&mut data)?;
    ///     s.dft_forward_in_place(&plan, &d)?;
    ///     drop(owner); // the transform may still be using the plan
    ///     Ok(())
    /// })?;
    /// # Ok(()) }
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::QueueMismatch`] if the plan belongs to another queue.
    pub fn plan<T: Scalar>(&'scope self, dft: &'scope Dft<T>) -> Result<Plan<'scope, T>> {
        if !self.queue.same_as(dft.queue()) {
            return Err(Error::QueueMismatch);
        }
        Ok(Plan { dft })
    }

    fn dft_run<T: Scalar>(
        &'scope self,
        plan: &Plan<'scope, T>,
        dir: i32,
        input: &UsmBuffer<Complex<T>>,
        output: Option<&UsmBuffer<Complex<T>>>,
    ) -> Result<Event<'scope>> {
        let count = plan.dft.len();
        for (name, b) in [("input", Some(input)), ("output", output)] {
            if let Some(b) = b.filter(|b| b.len() < count) {
                return Err(Error::InvalidArgument(format!(
                    "{name} has {} elements, the plan needs {count}",
                    b.len()
                )));
            }
        }
        let out_ptr = match output {
            Some(o) => {
                distinct(o, "output", input, "input")?;
                o.raw().cast::<c_void>()
            }
            None => std::ptr::null_mut(),
        };
        // SAFETY: the plan and both buffers are borrowed until the scope ends, which waits first;
        // lengths cover the plan; placement matches (checked by the callers); output is distinct
        // from input.
        let raw = unsafe {
            T::dft_compute_async(plan.dft.raw(), dir, input.raw().cast(), out_ptr, NO_DEPS, 0)
        };
        self.event(raw)
    }

    fn dft_in_place<T: Scalar>(
        &'scope self,
        plan: &Plan<'scope, T>,
        dir: i32,
        data: &Write<'scope, Complex<T>>,
    ) -> Result<Event<'scope>> {
        if plan.dft.placement() != Placement::InPlace {
            return Err(Error::InvalidArgument(
                "this plan is out-of-place; use forward/backward with two buffers".into(),
            ));
        }
        self.dft_run(plan, dir, &*data.buf, None)
    }

    fn dft_out_of_place<T: Scalar>(
        &'scope self,
        plan: &Plan<'scope, T>,
        dir: i32,
        input: &impl Operand<Complex<T>>,
        output: &Write<'scope, Complex<T>>,
    ) -> Result<Event<'scope>> {
        if plan.dft.placement() != Placement::NotInPlace {
            return Err(Error::InvalidArgument(
                "this plan is in-place; use forward_in_place/backward_in_place".into(),
            ));
        }
        self.dft_run(plan, dir, input.buffer(), Some(&*output.buf))
    }

    /// Forward transform of `data` in place (an in-place plan).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a wrong placement or a buffer shorter than the plan;
    /// submission errors as for [`Scope::gemm`].
    pub fn dft_forward_in_place<T: Scalar>(
        &'scope self,
        plan: &Plan<'scope, T>,
        data: &Write<'scope, Complex<T>>,
    ) -> Result<Event<'scope>> {
        self.dft_in_place(plan, 0, data)
    }

    /// Backward transform of `data` in place (an in-place plan).
    ///
    /// # Errors
    ///
    /// As [`Scope::dft_forward_in_place`].
    pub fn dft_backward_in_place<T: Scalar>(
        &'scope self,
        plan: &Plan<'scope, T>,
        data: &Write<'scope, Complex<T>>,
    ) -> Result<Event<'scope>> {
        self.dft_in_place(plan, 1, data)
    }

    /// Forward transform `input` into `output` (an out-of-place plan).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a wrong placement, a buffer shorter than the plan, or
    /// `output` being the same buffer as `input`; submission errors as for [`Scope::gemm`].
    pub fn dft_forward<T: Scalar>(
        &'scope self,
        plan: &Plan<'scope, T>,
        input: &impl Operand<Complex<T>>,
        output: &Write<'scope, Complex<T>>,
    ) -> Result<Event<'scope>> {
        self.dft_out_of_place(plan, 0, input, output)
    }

    /// Backward transform `input` into `output` (an out-of-place plan).
    ///
    /// # Errors
    ///
    /// As [`Scope::dft_forward`].
    pub fn dft_backward<T: Scalar>(
        &'scope self,
        plan: &Plan<'scope, T>,
        input: &impl Operand<Complex<T>>,
        output: &Write<'scope, Complex<T>>,
    ) -> Result<Event<'scope>> {
        self.dft_out_of_place(plan, 1, input, output)
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
