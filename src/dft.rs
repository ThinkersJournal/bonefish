//! Complex-to-complex discrete Fourier transforms through oneMKL.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use bonefish_sys as sys;

use crate::error::{Error, Result, last_message};
use crate::queue::Queue;
use crate::scalar::{Complex, Scalar};
use crate::usm::UsmBuffer;

/// Whether the transform overwrites its input.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Input and output are the same buffer.
    InPlace,
    /// Input is left alone and the result goes to a second buffer.
    NotInPlace,
}

/// A committed complex-to-complex plan for one shape and precision. Dropping it frees the plan.
///
/// Neither `Send` nor `Sync`. The oneAPI specification says "oneMKL objects in class-based APIs
/// may not be shared between concurrent host threads unless otherwise specified", and a DFT plan
/// is such an object; no exception was found, so the type does not claim one. (Moving a plan
/// to another thread is not "sharing" in that sentence, but the spec does not say it is fine
/// either, so that is not claimed.) To use one plan per thread, build one on each thread.
///
/// Checked by the compiler. The positive control passes (`Queue` is `Send + Sync`); the two
/// negative ones are pinned to `E0277` ("trait bound not satisfied") so they cannot pass for an
/// unrelated reason such as a typo:
///
/// ```
/// fn needs_send_sync<T: Send + Sync>() {}
/// needs_send_sync::<bonefish::Queue>();
/// needs_send_sync::<bonefish::UsmBuffer<f32>>();
/// ```
///
/// ```compile_fail,E0277
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<bonefish::Dft<f32>>();
/// ```
///
/// ```compile_fail,E0277
/// fn needs_send<T: Send>() {}
/// needs_send::<bonefish::Dft<f64>>();
/// ```
///
/// `forward` multiplies the result by `fwd_scale` and `backward` by `bwd_scale`; both default
/// to 1 in oneMKL, so an unnormalised round trip returns `n` times the input. Pass
/// `bwd_scale = 1.0 / n as f64` for a normalised inverse.
pub struct Dft<T: Scalar> {
    queue: Queue,
    handle: NonNull<c_void>,
    count: usize,
    placement: Placement,
    _marker: PhantomData<T>,
}

impl<T: Scalar> Dft<T> {
    /// Commit a plan over `lengths` (one entry per dimension, row-major as oneMKL defines it).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for no dimensions or a zero length; [`Error::Sycl`] /
    /// [`Error::Mkl`] if the device rejects the plan (for `f64`, a device without `fp64`).
    pub fn c2c(
        queue: &Queue,
        lengths: &[usize],
        placement: Placement,
        fwd_scale: f64,
        bwd_scale: f64,
    ) -> Result<Self> {
        if lengths.is_empty() || lengths.contains(&0) {
            return Err(Error::InvalidArgument(
                "a DFT needs at least one dimension and every length must be non-zero".into(),
            ));
        }
        let count = lengths
            .iter()
            .try_fold(1usize, |acc, l| acc.checked_mul(*l))
            .ok_or_else(|| Error::InvalidArgument("product of lengths overflows".into()))?;
        let dims: Vec<i64> = lengths
            .iter()
            .map(|l| {
                i64::try_from(*l)
                    .map_err(|_| Error::InvalidArgument(format!("length {l} exceeds i64")))
            })
            .collect::<Result<_>>()?;
        // SAFETY: `dims` is valid for `dims.len()` entries for the duration of the call.
        let raw = unsafe {
            sys::bf_dft_c2c_new(
                queue.raw(),
                T::PREC,
                dims.len() as i64,
                dims.as_ptr(),
                i32::from(placement == Placement::InPlace),
                fwd_scale,
                bwd_scale,
            )
        };
        let handle = NonNull::new(raw).ok_or_else(|| {
            if sys::SYCL_AVAILABLE {
                Error::Mkl(last_message())
            } else {
                Error::Unavailable(last_message())
            }
        })?;
        Ok(Dft {
            queue: queue.clone(),
            handle,
            count,
            placement,
            _marker: PhantomData,
        })
    }

    /// Number of complex elements the plan transforms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Always `false`: a plan has at least one element.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    fn check(&self, buf: &UsmBuffer<Complex<T>>, name: &str) -> Result<()> {
        if !self.queue.same_as(buf.queue()) {
            return Err(Error::QueueMismatch);
        }
        if buf.len() < self.count {
            return Err(Error::InvalidArgument(format!(
                "{name} has {} elements, the plan needs {}",
                buf.len(),
                self.count
            )));
        }
        Ok(())
    }

    fn run_in_place(&self, dir: i32, data: &mut UsmBuffer<Complex<T>>) -> Result<()> {
        if self.placement != Placement::InPlace {
            return Err(Error::InvalidArgument(
                "this plan is out-of-place; use forward/backward with two buffers".into(),
            ));
        }
        self.check(data, "data")?;
        // SAFETY: the plan and buffer share a queue; the buffer holds at least `count` complex
        // elements; `data` is exclusively borrowed.
        let rc = unsafe {
            sys::bf_dft_compute(
                self.handle.as_ptr(),
                dir,
                data.raw().cast(),
                std::ptr::null_mut(),
            )
        };
        Queue::status(rc)
    }

    fn run_out_of_place(
        &self,
        dir: i32,
        input: &UsmBuffer<Complex<T>>,
        output: &mut UsmBuffer<Complex<T>>,
    ) -> Result<()> {
        if self.placement != Placement::NotInPlace {
            return Err(Error::InvalidArgument(
                "this plan is in-place; use forward_in_place/backward_in_place".into(),
            ));
        }
        self.check(input, "input")?;
        self.check(output, "output")?;
        // SAFETY: as `run_in_place`; `output` is exclusively borrowed so it cannot alias `input`.
        let rc = unsafe {
            sys::bf_dft_compute(
                self.handle.as_ptr(),
                dir,
                input.raw().cast(),
                output.raw().cast(),
            )
        };
        Queue::status(rc)
    }

    /// Forward transform of an in-place plan.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the plan is out-of-place or the buffer is too short;
    /// [`Error::QueueMismatch`]; device errors.
    pub fn forward_in_place(&self, data: &mut UsmBuffer<Complex<T>>) -> Result<()> {
        self.run_in_place(0, data)
    }

    /// Backward (inverse, unscaled unless `bwd_scale` says otherwise) transform of an in-place
    /// plan.
    ///
    /// # Errors
    ///
    /// As [`Dft::forward_in_place`].
    pub fn backward_in_place(&self, data: &mut UsmBuffer<Complex<T>>) -> Result<()> {
        self.run_in_place(1, data)
    }

    /// Forward transform of an out-of-place plan; `input` is not modified.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the plan is in-place or a buffer is too short;
    /// [`Error::QueueMismatch`]; device errors.
    pub fn forward(
        &self,
        input: &UsmBuffer<Complex<T>>,
        output: &mut UsmBuffer<Complex<T>>,
    ) -> Result<()> {
        self.run_out_of_place(0, input, output)
    }

    /// Backward transform of an out-of-place plan.
    ///
    /// # Errors
    ///
    /// As [`Dft::forward`].
    pub fn backward(
        &self,
        input: &UsmBuffer<Complex<T>>,
        output: &mut UsmBuffer<Complex<T>>,
    ) -> Result<()> {
        self.run_out_of_place(1, input, output)
    }
}

impl<T: Scalar> Drop for Dft<T> {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `bf_dft_c2c_new` and is freed once; the plan owns a copy of
        // the queue in the shim, and `self.queue` keeps ours alive until after this runs.
        unsafe { sys::bf_dft_free(self.handle.as_ptr()) }
    }
}
