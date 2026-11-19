//! Raw FFI to Bonefish's C shim over oneMKL's SYCL interface. See the `bonefish` crate for the
//! safe API.
//!
//! The shim is only built when Intel's DPC++ compiler is found at build time; otherwise this
//! crate is empty and `cfg(bonefish_sycl)` is not set. Handles are opaque pointers; status codes
//! are `0` ok, `1` SYCL error, `2` oneMKL error, `3` other, with the message from
//! `bf_last_error`.

#[cfg(bonefish_sycl)]
mod ffi;

#[cfg(bonefish_sycl)]
pub use ffi::*;

/// Whether the shim was built into this crate.
pub const SYCL_AVAILABLE: bool = cfg!(bonefish_sycl);

/// Stand-ins used when the shim was not built (no oneAPI at build time): same signatures as the
/// real foreign functions, so everything above this crate compiles identically either way.
/// Constructors return null, status-returning calls return [`STATUS_UNAVAILABLE`], and
/// `bf_last_error` explains. The real/stub pair is checked by the compiler in the two CI
/// configurations (with and without oneAPI), because the safe layer calls whichever is active.
// The stand-ins must be `unsafe` to match the real foreign declarations (callers wrap them in
// `unsafe` either way), and they take the same long argument lists, so these two lints fire on
// them for reasons that do not apply: they dereference nothing and have no safety contract.
#[cfg(not(bonefish_sycl))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
mod stub;

#[cfg(not(bonefish_sycl))]
pub use stub::*;
