//! Real-to-complex and complex-to-real DFT (R3b).
//!
//! [`RealDft`] transforms an array of real numbers into its **half spectrum** and back. A real
//! signal's spectrum is conjugate-symmetric, so only part of it is stored: the last dimension of
//! the complex side has `n_last / 2 + 1` entries instead of `n_last`.
//!
//! # Layout
//!
//! Both arrays are dense and row-major (the last dimension is contiguous):
//!
//! | side | shape |
//! |---|---|
//! | real | `lengths[0] x ... x lengths[d-1]` |
//! | complex | `lengths[0] x ... x lengths[d-2] x (lengths[d-1] / 2 + 1)` |
//!
//! The shim sets these strides **explicitly**. oneMKL's default real layout is the one for an
//! unbatched *in-place* transform, with minimal padding in the real domain (documented: Intel's
//! oneMKL DPC++ Developer Reference 2026.0, "Configuring data layouts", sections "Configuring
//! strides in forward and backward domains" and "Data layout requirements"), and it applies even
//! out of place; relying on it read a dense array with the wrong
//! row stride (measured, CPU device, oneAPI 2026.1: a delta at element (1, 0) of a 4 x 6 array was
//! treated as padding and ignored). Out-of-place only; in-place real transforms need padded
//! storage and are not offered.
//!
//! The layout is checked against a naive DFT in 1-D, 2-D and 3-D on the CPU/OpenCL device. Whether
//! a GPU produces the same layout is **UNVERIFIED**.
//!
//! # Scaling
//!
//! `forward` multiplies by `fwd_scale` and `backward` by `bwd_scale`; both default to 1 in
//! oneMKL, so an unnormalised round trip returns `prod(lengths)` times the input. Pass
//! `bwd_scale = 1 / prod(lengths)` for an exact round trip.
//!
//! # Threads
//!
//! Like [`Dft`](crate::Dft), a `RealDft` is neither `Send` nor `Sync` (oneMKL says class-based
//! objects may not be shared between concurrent host threads). Checked by the compiler, with the
//! positive control first:
//!
//! ```
//! fn needs_send_sync<T: Send + Sync>() {}
//! needs_send_sync::<bonefish::Queue>();
//! ```
//!
//! ```compile_fail,E0277
//! fn needs_sync<T: Sync>() {}
//! needs_sync::<bonefish::RealDft<f32>>();
//! ```
//!
//! ```compile_fail,E0277
//! fn needs_send<T: Send>() {}
//! needs_send::<bonefish::RealDft<f64>>();
//! ```

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use bonefish_sys as sys;

use crate::error::{Error, Result, last_message};
use crate::queue::Queue;
use crate::scalar::{Complex, Scalar};
use crate::usm::UsmBuffer;

/// A committed real-to-complex / complex-to-real DFT plan of `f32` or `f64`.
pub struct RealDft<T: Scalar> {
    queue: Queue,
    handle: NonNull<c_void>,
    lengths: Vec<usize>,
    real_len: usize,
    complex_len: usize,
    _marker: PhantomData<T>,
}

impl<T: Scalar> RealDft<T> {
    /// Plan a real DFT over the given dimensions (last dimension contiguous).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for no dimensions, a zero length, or sizes that overflow;
    /// [`Error::Mkl`] if oneMKL cannot build the plan; [`Error::Unavailable`] without oneAPI.
    pub fn new(queue: &Queue, lengths: &[usize], fwd_scale: f64, bwd_scale: f64) -> Result<Self> {
        if lengths.is_empty() || lengths.contains(&0) {
            return Err(Error::InvalidArgument(
                "a DFT needs at least one dimension and every length must be non-zero".into(),
            ));
        }
        let overflow = || Error::InvalidArgument("product of lengths overflows".into());
        let real_len = lengths
            .iter()
            .try_fold(1usize, |acc, l| acc.checked_mul(*l))
            .ok_or_else(overflow)?;
        let (last, leading) = lengths.split_last().expect("non-empty");
        let complex_len = leading
            .iter()
            .try_fold(last / 2 + 1, |acc, l| acc.checked_mul(*l))
            .ok_or_else(overflow)?;
        let dims: Vec<i64> = lengths
            .iter()
            .map(|l| {
                i64::try_from(*l)
                    .map_err(|_| Error::InvalidArgument(format!("length {l} exceeds i64")))
            })
            .collect::<Result<_>>()?;
        // SAFETY: `dims` is valid for `dims.len()` reads; a null return is handled below.
        let raw = unsafe {
            sys::bf_dftr_new(
                queue.raw(),
                T::PREC,
                dims.len() as i64,
                dims.as_ptr(),
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
        Ok(RealDft {
            queue: queue.clone(),
            handle,
            lengths: lengths.to_vec(),
            real_len,
            complex_len,
            _marker: PhantomData,
        })
    }

    /// The dimensions of the real side.
    #[must_use]
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Number of real elements: `prod(lengths)`.
    #[must_use]
    pub fn real_len(&self) -> usize {
        self.real_len
    }

    /// Number of complex elements in the half spectrum: `prod(lengths[..d-1]) * (lengths[d-1] / 2 + 1)`.
    #[must_use]
    pub fn complex_len(&self) -> usize {
        self.complex_len
    }

    fn check_queue(&self, queue: &Queue) -> Result<()> {
        if self.queue.same_as(queue) {
            Ok(())
        } else {
            Err(Error::QueueMismatch)
        }
    }

    fn check_len(name: &str, have: usize, need: usize) -> Result<()> {
        if have < need {
            return Err(Error::InvalidArgument(format!(
                "{name} has {have} elements, the plan needs {need}"
            )));
        }
        Ok(())
    }

    /// Real input to half-spectrum output. Blocking.
    ///
    /// # Errors
    ///
    /// [`Error::QueueMismatch`] for a buffer from another queue; [`Error::InvalidArgument`] if a
    /// buffer is shorter than the plan needs; device errors.
    pub fn forward(&self, input: &UsmBuffer<T>, output: &mut UsmBuffer<Complex<T>>) -> Result<()> {
        self.check_queue(input.queue())?;
        self.check_queue(output.queue())?;
        Self::check_len("input", input.len(), self.real_len)?;
        Self::check_len("output", output.len(), self.complex_len)?;
        // SAFETY: both buffers are USM of this plan's queue and cover the plan's sizes (checked
        // above); `output` is exclusively borrowed so it cannot alias `input`; the call blocks.
        let rc = unsafe {
            sys::bf_dftr_compute(
                self.handle.as_ptr(),
                0,
                input.raw().cast(),
                output.raw().cast(),
            )
        };
        Queue::status(rc)
    }

    /// Half-spectrum input to real output. Blocking.
    ///
    /// # Errors
    ///
    /// As [`RealDft::forward`].
    pub fn backward(&self, input: &UsmBuffer<Complex<T>>, output: &mut UsmBuffer<T>) -> Result<()> {
        self.check_queue(input.queue())?;
        self.check_queue(output.queue())?;
        Self::check_len("input", input.len(), self.complex_len)?;
        Self::check_len("output", output.len(), self.real_len)?;
        // SAFETY: as `forward`.
        let rc = unsafe {
            sys::bf_dftr_compute(
                self.handle.as_ptr(),
                1,
                input.raw().cast(),
                output.raw().cast(),
            )
        };
        Queue::status(rc)
    }
}

impl<T: Scalar> Drop for RealDft<T> {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `bf_dftr_new` and is freed exactly once, here, while the
        // queue it was made on is still alive (this struct holds a clone of it).
        unsafe { sys::bf_dftr_free(self.handle.as_ptr()) }
    }
}

impl<T: Scalar> std::fmt::Debug for RealDft<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealDft")
            .field("lengths", &self.lengths)
            .finish()
    }
}
