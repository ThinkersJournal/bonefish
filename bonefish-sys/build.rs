//! Builds `shim/*.cpp` into a shared library with Intel's DPC++ compiler (`icpx -fsycl`) and
//! links it. Without oneAPI the crate still builds (the `bonefish_sycl` cfg is simply not set), so
//! `cargo test --all-features` works anywhere; set `BONEFISH_REQUIRE_SYCL=1` to turn "no oneAPI
//! found" into a hard error (the oneAPI CI job does, so it can never pass by skipping), and
//! `BONEFISH_NO_SYCL=1` skips the shim on purpose.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(bonefish_sycl)");
    println!("cargo::rerun-if-changed=shim");
    println!("cargo::rerun-if-changed=build.rs");
    for v in [
        "BONEFISH_ICPX",
        "BONEFISH_REQUIRE_SYCL",
        "BONEFISH_NO_SYCL",
        "ONEAPI_ROOT",
        "MKLROOT",
        "PATH",
    ] {
        println!("cargo::rerun-if-env-changed={v}");
    }
    let require = env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1");

    // Deliberately build without the shim (the crate then uses its failing stand-ins). Useful to
    // test that path on a machine that has oneAPI, and for users who do not want it built.
    if env::var("BONEFISH_NO_SYCL").is_ok_and(|v| v == "1") {
        assert!(
            !require,
            "BONEFISH_NO_SYCL=1 contradicts BONEFISH_REQUIRE_SYCL=1"
        );
        println!("cargo::warning=BONEFISH_NO_SYCL=1: building without SYCL support");
        return;
    }

    let Some((icpx, root)) = find_icpx() else {
        let msg = "oneAPI DPC++ compiler (icpx) not found: set BONEFISH_ICPX or ONEAPI_ROOT, \
                   or run oneAPI's setvars; building without SYCL support";
        if require {
            panic!("BONEFISH_REQUIRE_SYCL=1 but {msg}");
        }
        println!("cargo::warning={msg}");
        return;
    };

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let windows = env::var("CARGO_CFG_TARGET_OS").is_ok_and(|o| o == "windows");
    let lib = if windows {
        out.join("bonefish_shim.dll")
    } else {
        out.join("libbonefish_shim.so")
    };

    let mut cmd = Command::new(&icpx);
    cmd.args(["-fsycl", "-qmkl=parallel", "-shared", "-O2"]);
    if !windows {
        cmd.arg("-fPIC");
    }
    cmd.args([
        "shim/shim.cpp",
        "shim/dft.cpp",
        "shim/blas1.cpp",
        "shim/blas1_async.cpp",
    ])
    .arg("-o")
    .arg(&lib);
    if env::var_os("MKLROOT").is_none()
        && let Some(root) = root
    {
        cmd.env("MKLROOT", root.join("mkl").join("latest"));
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", icpx.display()));
    assert!(status.success(), "icpx failed building the shim ({status})");

    println!("cargo::rustc-link-search=native={}", out.display());
    println!("cargo::rustc-link-lib=dylib=bonefish_shim");
    println!("cargo::rustc-cfg=bonefish_sycl");
}

/// Returns the compiler and, when known, the oneAPI root it came from.
fn find_icpx() -> Option<(PathBuf, Option<PathBuf>)> {
    let exe = if cfg!(windows) { "icpx.exe" } else { "icpx" };
    if let Ok(p) = env::var("BONEFISH_ICPX") {
        return Some((PathBuf::from(p), None));
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(r) = env::var("ONEAPI_ROOT") {
        roots.push(r.into());
    }
    if cfg!(windows) {
        roots.push(r"C:\Program Files (x86)\Intel\oneAPI".into());
    } else {
        roots.push("/opt/intel/oneapi".into());
    }
    for root in roots {
        let found = [
            root.join("compiler").join("latest").join("bin").join(exe),
            root.join("compiler")
                .join("latest")
                .join("bin")
                .join("compiler")
                .join(exe),
        ]
        .into_iter()
        .find(|p| p.is_file());
        if let Some(p) = found {
            return Some((p, Some(root)));
        }
    }
    // Last resort: whatever `icpx` is on PATH (after setvars).
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|d| d.join(exe))
            .find(|p| p.is_file())
            .map(|p| (p, None))
    })
}
