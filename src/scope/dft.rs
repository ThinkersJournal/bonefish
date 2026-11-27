// SPDX-License-Identifier: MIT OR Apache-2.0
//! DFT on [`Scope`]: plan binding and the forward and backward transforms. Split out of
//! `scope.rs` (R15).

use super::*;

/// A DFT plan bound for use inside a scope. Holds a shared borrow of the plan until the scope
/// ends, so the plan cannot be dropped while a transform that uses it is still running.
pub struct Plan<'scope, T: Scalar> {
    dft: &'scope Dft<T>,
}

/// A real DFT plan bound for use inside a scope, like [`Plan`]: it holds a shared borrow of the
/// plan until the scope ends.
pub struct RealPlan<'scope, T: Scalar> {
    dft: &'scope RealDft<T>,
}

/// DFT: bind a plan, then submit transforms. Same ordering as the other scope operations.
impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
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

    /// Bind a real DFT plan for use in this scope; as [`Scope::plan`] for the complex one.
    ///
    /// # Errors
    ///
    /// [`Error::QueueMismatch`] if the plan belongs to another queue.
    pub fn real_plan<T: Scalar>(
        &'scope self,
        dft: &'scope RealDft<T>,
    ) -> Result<RealPlan<'scope, T>> {
        if !self.queue.same_as(dft.queue()) {
            return Err(Error::QueueMismatch);
        }
        Ok(RealPlan { dft })
    }

    fn rdft_run<A: Element, B: Element, T: Scalar>(
        &'scope self,
        plan: &RealPlan<'scope, T>,
        dir: i32,
        (input, in_need): (&UsmBuffer<A>, usize),
        (output, out_need): (&UsmBuffer<B>, usize),
    ) -> Result<Event<'scope>> {
        for (name, have, need) in [
            ("input", input.len(), in_need),
            ("output", output.len(), out_need),
        ] {
            if have < need {
                return Err(Error::InvalidArgument(format!(
                    "{name} has {have} elements, the plan needs {need}"
                )));
            }
        }
        distinct(output, "output", input, "input")?;
        // SAFETY: the plan and both buffers are borrowed until the scope ends, which waits first;
        // lengths cover the plan (checked above); the element types are the plan's real and
        // half-spectrum complex types for `dir` (fixed by the two callers); output is distinct
        // from input.
        let raw = unsafe {
            sys::bf_dftr_compute_async(
                plan.dft.raw(),
                dir,
                input.raw().cast(),
                output.raw().cast(),
                NO_DEPS,
                0,
            )
        };
        self.event(raw)
    }

    /// Real input to half-spectrum output (the plan is always out-of-place; layout as
    /// [`RealDft`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a buffer shorter than the plan or `output` being the same
    /// buffer as `input`; submission errors as for [`Scope::gemm`].
    pub fn rdft_forward<T: Scalar>(
        &'scope self,
        plan: &RealPlan<'scope, T>,
        input: &impl Operand<T>,
        output: &Write<'scope, Complex<T>>,
    ) -> Result<Event<'scope>> {
        let (r, c) = (plan.dft.real_len(), plan.dft.complex_len());
        self.rdft_run(plan, 0, (input.buffer(), r), (&*output.buf, c))
    }

    /// Half-spectrum input to real output.
    ///
    /// # Errors
    ///
    /// As [`Scope::rdft_forward`].
    pub fn rdft_backward<T: Scalar>(
        &'scope self,
        plan: &RealPlan<'scope, T>,
        input: &impl Operand<Complex<T>>,
        output: &Write<'scope, T>,
    ) -> Result<Event<'scope>> {
        let (r, c) = (plan.dft.real_len(), plan.dft.complex_len());
        self.rdft_run(plan, 1, (input.buffer(), c), (&*output.buf, r))
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
