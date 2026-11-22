//! The shim DLL a test process loads must be the one this build produced: a stale copy earlier on
//! the loader path once hung `tests/level3.rs` for hours. Prints the loaded path and timestamp.

use bonefish_sys::{BUILT_SHIM, loaded_shim_path, shim_is_fresh};
use std::path::Path;

#[test]
fn loaded_shim_is_the_one_just_built() {
    // Touch a real shim symbol so the linker keeps the import and Windows loads the DLL.
    #[cfg(bonefish_sycl)]
    let _ = unsafe { bonefish_sys::bf_last_error() };
    let (Some(built), Some(loaded)) = (BUILT_SHIM, loaded_shim_path()) else {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1") || cfg!(not(windows)),
            "BONEFISH_REQUIRE_SYCL=1 but no shim path to check"
        );
        eprintln!("SKIP: no shim built or loaded-path query unsupported here");
        return;
    };
    let stamp = std::fs::metadata(&loaded).and_then(|m| m.modified()).ok();
    eprintln!("shim loaded from {} (modified {stamp:?})", loaded.display());
    shim_is_fresh(Path::new(built), &loaded).unwrap();
}

#[test]
fn a_stale_copy_is_refused_by_name() {
    // Seeded: a different existing file stands in for a stale DLL earlier on PATH.
    let me = std::env::current_exe().unwrap();
    let err = shim_is_fresh(&me, Path::new("Cargo.toml")).unwrap_err();
    assert!(err.starts_with("STALE SHIM"), "{err}");
    assert!(shim_is_fresh(&me, &me).is_ok());
}
