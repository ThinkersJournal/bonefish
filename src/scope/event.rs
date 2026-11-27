// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`Event`] and [`DeviceTime`]: handles to submitted device work (moved out of `scope.rs`, R15).

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use bonefish_sys as sys;

use crate::error::{Error, Result, check};

/// A handle to submitted device work. It cannot outlive its scope. Dropping it only releases the
/// handle (the work still completes and the scope still waits for it), so leaking one costs
/// memory, never safety.
pub struct Event<'scope> {
    pub(super) raw: NonNull<c_void>,
    /// Its queue was created with profiling.
    profiling: bool,
    _scope: PhantomData<&'scope mut &'scope ()>,
}

impl Event<'_> {
    /// Wrap a live event handle from a `*_async` call.
    pub(super) fn new<'scope>(raw: NonNull<c_void>, profiling: bool) -> Event<'scope> {
        Event {
            raw,
            profiling,
            _scope: PhantomData,
        }
    }
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
    /// ([`Queue::new_profiling`](crate::Queue::new_profiling)): an error, never a zero. Otherwise the asynchronous error, if the
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
