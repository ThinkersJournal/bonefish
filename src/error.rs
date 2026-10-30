//! Errors from the safe layer.

use std::ffi::CStr;
use std::fmt;

/// What went wrong. Status codes from the C shim map to the first four variants; the rest are
/// caught in Rust before anything reaches the device.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Built without oneAPI (`icpx` was not found at build time), so there is no SYCL runtime to
    /// call. Carries the shim's message.
    Unavailable(String),
    /// A SYCL runtime exception (no such device, allocation failure, device lacks `fp64`, ...).
    Sycl(String),
    /// An oneMKL exception, typically an invalid argument it detected itself.
    Mkl(String),
    /// Any other C++ exception.
    Other(String),
    /// Rejected in Rust before the call: a dimension, leading dimension or buffer length that
    /// would have read or written out of bounds, or an operation not valid for this object.
    InvalidArgument(String),
    /// A buffer was allocated from a different queue than the one the operation was given. USM
    /// pointers are only valid in the context that allocated them.
    QueueMismatch,
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unavailable(m) => write!(f, "SYCL unavailable: {m}"),
            Error::Sycl(m) => write!(f, "SYCL error: {m}"),
            Error::Mkl(m) => write!(f, "oneMKL error: {m}"),
            Error::Other(m) => write!(f, "error: {m}"),
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::QueueMismatch => {
                write!(
                    f,
                    "buffer was allocated from a different queue than the operation's"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// The shim's message for the last failure on this thread.
pub(crate) fn last_message() -> String {
    // SAFETY: `bf_last_error` returns a NUL-terminated string that stays valid until the next
    // failing call on this thread; it is copied immediately.
    unsafe { CStr::from_ptr(bonefish_sys::bf_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// Map a shim status code to a result. `0` is success.
pub(crate) fn check(status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        1 => Err(Error::Sycl(last_message())),
        2 => Err(Error::Mkl(last_message())),
        4 => Err(Error::Unavailable(last_message())),
        _ => Err(Error::Other(last_message())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `check` consults the shim only for non-zero codes, and the shim (or its stand-in) always
    // answers `bf_last_error`, so these hold with or without oneAPI.
    #[test]
    fn zero_is_ok_and_codes_map_to_variants() {
        assert_eq!(check(0), Ok(()));
        assert!(matches!(check(1), Err(Error::Sycl(_))));
        assert!(matches!(check(2), Err(Error::Mkl(_))));
        assert!(matches!(check(4), Err(Error::Unavailable(_))));
        assert!(matches!(check(3), Err(Error::Other(_))));
        assert!(matches!(check(99), Err(Error::Other(_))));
    }

    #[test]
    fn display_names_the_kind() {
        assert!(Error::QueueMismatch.to_string().contains("different queue"));
        assert!(
            Error::InvalidArgument("lda".into())
                .to_string()
                .starts_with("invalid argument")
        );
    }
}
