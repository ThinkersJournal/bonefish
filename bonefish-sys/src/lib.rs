// SPDX-License-Identifier: MIT OR Apache-2.0
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

/// The shim library this build produced (`None` without oneAPI). Tests compare it with
/// [`loaded_shim_path`] so a stale copy earlier on the loader path cannot pass (or hang) unseen.
pub const BUILT_SHIM: Option<&str> = option_env!("BONEFISH_SHIM_BUILT");

/// Full path of the shim library this process actually loaded (Windows only; `None` elsewhere or
/// without the shim).
#[cfg(all(bonefish_sycl, windows))]
pub fn loaded_shim_path() -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut core::ffi::c_void;
        fn GetModuleFileNameW(m: *mut core::ffi::c_void, buf: *mut u16, n: u32) -> u32;
    }
    let name: Vec<u16> = "bonefish_shim.dll\0".encode_utf16().collect();
    // SAFETY: `name` is NUL-terminated; the buffer length passed is its real length.
    unsafe {
        let m = GetModuleHandleW(name.as_ptr());
        if m.is_null() {
            return None;
        }
        let mut buf = [0u16; 1024];
        let n = GetModuleFileNameW(m, buf.as_mut_ptr(), 1024) as usize;
        (n > 0 && n < 1024).then(|| std::ffi::OsString::from_wide(&buf[..n]).into())
    }
}

/// See the Windows version; this platform does not report the loaded path.
#[cfg(not(all(bonefish_sycl, windows)))]
pub fn loaded_shim_path() -> Option<std::path::PathBuf> {
    None
}

/// `Err` (naming both paths and the loaded file's timestamp) unless `loaded` is the file `built`.
///
/// # Errors
///
/// When the two paths name different files.
pub fn shim_is_fresh(built: &std::path::Path, loaded: &std::path::Path) -> Result<(), String> {
    let same = match (std::fs::canonicalize(built), std::fs::canonicalize(loaded)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if same {
        return Ok(());
    }
    let stamp = std::fs::metadata(loaded).and_then(|m| m.modified()).ok();
    Err(format!(
        "STALE SHIM: loaded {} (modified {stamp:?}) but this build produced {}",
        loaded.display(),
        built.display()
    ))
}

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
