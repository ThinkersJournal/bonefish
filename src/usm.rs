//! Unified shared memory owned by a queue.

use std::marker::PhantomData;
use std::ptr::NonNull;

use bonefish_sys as sys;

use crate::error::{Error, Result, check, last_message};
use crate::queue::Queue;
use crate::scalar::Element;

/// Where an allocation lives.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UsmKind {
    /// Shared: visible to host and device. Host access is allowed.
    Shared,
    /// Device only. The host reaches it through copies, never through a slice.
    Device,
    /// Host memory the device can read. Host access is allowed.
    Host,
}

impl UsmKind {
    fn code(self) -> i32 {
        match self {
            UsmKind::Shared => 0,
            UsmKind::Device => 1,
            UsmKind::Host => 2,
        }
    }

    fn host_accessible(self) -> bool {
        !matches!(self, UsmKind::Device)
    }
}

/// A typed array in USM, freed with the queue's context when dropped.
///
/// Every operation in this crate is blocking, so nothing is ever in flight when you touch a
/// `Shared` or `Host` buffer from the host.
pub struct UsmBuffer<T: Element> {
    queue: Queue,
    ptr: NonNull<T>,
    len: usize,
    kind: UsmKind,
    _marker: PhantomData<T>,
}

// SAFETY: the buffer is a uniquely owned USM allocation of plain `Element` data plus a `Queue`
// (itself `Send + Sync`, see there). Moving it to another thread moves ownership of the pointer;
// sharing `&UsmBuffer` only allows reads (`to_vec`, `as_slice`) and passing it as a read-only
// operand, and every mutation takes `&mut self`.
unsafe impl<T: Element> Send for UsmBuffer<T> {}
// SAFETY: as above.
unsafe impl<T: Element> Sync for UsmBuffer<T> {}

impl<T: Element> UsmBuffer<T> {
    fn alloc(queue: &Queue, kind: UsmKind, len: usize) -> Result<Self> {
        // At least one element, so the pointer is never null for an empty buffer.
        let bytes = len
            .max(1)
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|b| i64::try_from(b).ok())
            .ok_or_else(|| {
                Error::InvalidArgument(format!("allocation of {len} elements overflows"))
            })?;
        // SAFETY: plain call; null is handled.
        let raw = unsafe { sys::bf_malloc(queue.raw(), kind.code(), bytes) };
        let ptr = NonNull::new(raw.cast::<T>()).ok_or_else(|| {
            if sys::SYCL_AVAILABLE {
                Error::Sycl(last_message())
            } else {
                Error::Unavailable(last_message())
            }
        })?;
        Ok(UsmBuffer {
            queue: queue.clone(),
            ptr,
            len,
            kind,
            _marker: PhantomData,
        })
    }

    /// A buffer holding a copy of `data`.
    ///
    /// # Errors
    ///
    /// [`Error::Sycl`] if the allocation or copy fails.
    pub fn from_slice(queue: &Queue, kind: UsmKind, data: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(queue, kind, data.len())?;
        buf.copy_from_slice(data)?;
        Ok(buf)
    }

    /// A buffer of `len` zeros.
    ///
    /// # Errors
    ///
    /// As [`UsmBuffer::from_slice`].
    pub fn zeroed(queue: &Queue, kind: UsmKind, len: usize) -> Result<Self> {
        let zeros = vec![T::ZERO; len];
        Self::from_slice(queue, kind, &zeros)
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Where the memory lives.
    #[must_use]
    pub fn kind(&self) -> UsmKind {
        self.kind
    }

    /// The queue that allocated this buffer.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Overwrite the contents with `data`, which must be exactly [`len`](Self::len) elements.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on a length mismatch; [`Error::Sycl`] if the copy fails.
    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(Error::InvalidArgument(format!(
                "copy of {} elements into a buffer of {}",
                data.len(),
                self.len
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::size_of_val(data) as i64;
        // SAFETY: `ptr` owns `len` elements of USM; `data` is a valid slice of the same length;
        // the shim's memcpy is blocking.
        check(unsafe {
            sys::bf_memcpy(
                self.queue.raw(),
                self.ptr.as_ptr().cast(),
                data.as_ptr().cast(),
                bytes,
            )
        })
    }

    /// Copy the contents out to a `Vec`. Works for every [`UsmKind`].
    ///
    /// # Errors
    ///
    /// [`Error::Sycl`] if the copy fails.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut out = vec![T::ZERO; self.len];
        if self.len == 0 {
            return Ok(out);
        }
        let bytes = (self.len * std::mem::size_of::<T>()) as i64;
        // SAFETY: `out` has `len` elements; `ptr` owns `len` elements of USM; blocking copy.
        check(unsafe {
            sys::bf_memcpy(
                self.queue.raw(),
                out.as_mut_ptr().cast(),
                self.ptr.as_ptr().cast(),
                bytes,
            )
        })?;
        Ok(out)
    }

    /// The contents as a slice, for `Shared` and `Host` buffers; `None` for `Device`.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[T]> {
        // SAFETY: host-accessible USM holding `len` initialised elements (zero-filled or copied
        // on creation); nothing runs on the device concurrently because every operation blocks.
        self.kind
            .host_accessible()
            .then(|| unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) })
    }

    /// Mutable slice; `None` for `Device`.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        // SAFETY: as `as_slice`, and `&mut self` makes the access exclusive.
        self.kind
            .host_accessible()
            .then(|| unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
    }

    pub(crate) fn raw(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T: Element> Drop for UsmBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `bf_malloc` on this queue, which is still alive because the
        // buffer holds a clone of it.
        unsafe { sys::bf_free(self.queue.raw(), self.ptr.as_ptr().cast()) }
    }
}

impl<T: Element> std::fmt::Debug for UsmBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsmBuffer")
            .field("len", &self.len)
            .field("kind", &self.kind)
            .finish()
    }
}
