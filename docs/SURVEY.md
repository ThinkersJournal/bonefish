# Bonefish survey and design note (R1)

Status: options and a recommendation; **no code**. Author: Bonefish lane, 2026-10-30. Decisions marked **[PM]** or **[CireSnave]** are not mine to make.

Measured on the laptop (oneAPI 2026.1 at `C:\Program Files (x86)\Intel\oneAPI`) and against `origin/main` of the named repos on 2026-10-30. Anything not so measured is marked UNVERIFIED.

## 1. What is on the box (measured)

- DPC++ compiler: `icpx.exe`, "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0 (2026.1.0.20260617)", target `x86_64-pc-windows-msvc`.
- oneMKL SYCL API: `mkl/latest/include/oneapi/mkl/{blas,dft,lapack,rng,spblas,stats,vm}`. Every entry point takes a `sycl::queue&` and returns a `sycl::event`, e.g. `gemm(sycl::queue&, transpose, transpose, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, [compute_mode], const std::vector<sycl::event>& deps)` (`blas/usm_decls.hpp:22`). Instantiated for f32, f64, complex, `sycl::half`, `bfloat16` and int8 inputs (`usm_decls.hpp:38-47`). It is C++ only: no `extern "C"` form, so any Rust binding needs a shim we write.
- Libraries: `mkl/latest/lib/mkl_sycl_{blas,dft,lapack,rng,sparse,stats,vm,data_fitting}_dll.lib` and `bin/mkl_sycl_*.6.dll`.
- **No GPU is visible on this laptop.** `sycl-ls.exe` lists one device, `[opencl:cpu] AMD Ryzen 9 7940HX`. GPU results need the desktop A770 or a self-hosted runner (portfolio rules 5.1g/5.1h); GitHub-hosted CI can only exercise the CPU/OpenCL device.
- Prior art on crates.io: `sycl-rs`, `oneapi-rs`, `sycl-rs-sys` 0.1.0 (MIT OR Apache-2.0, rust-version 1.85), repo `oneapi-src/oneapi-rs` (Apache-2.0 per the forge, pushed 2026-10-29). Its README says experimental, "API may change without notice", **tested on Linux + oneAPI 2026.1 only**; it offers a SYCL runtime (device listing, USM, runtime kernel compile, launch). I found no oneMKL layer in it, but read only the README, not the source. Also `oxicuda-levelzero` (Apache-2.0, third party, pure Rust via libloading) and `ironaccelerator-levelzero` (AGPL-3.0, unusable for us).
- `onemkl-sys` / `onemkl` 0.2.0 exist (CPU C interface). PR #1 in `ciresnave/onemkl` withdraws their "SYCL out of scope" line.

## 2. Options for the oneMKL-on-GPU path

**A. Own C++ shim, own runtime layer.** `bonefish-sys` compiles a small `extern "C"` shim over `oneapi::mkl::*` with `icpx -fsycl` (from `build.rs`), exposing opaque queue/event/USM handles; `bonefish` wraps them safely. Full control, Windows and Linux, nothing experimental underneath. Cost: we write queue/USM/event management (duplicating part of `sycl-rs`) and own the C++/Rust ABI and `build.rs` toolchain discovery.

**B. Build on Intel's `sycl-rs` for queue/USM, add only the oneMKL shim.** Less code, aligned with Intel's own bindings. Risks: 0.1.0 and explicitly unstable; Windows untested by them; our C++ shim needs the real `sycl::queue`, which only works if their FFI exposes a stable handle to it (UNVERIFIED, source not read). Couples our API to theirs.

**C. Level Zero directly (Rust, `libloading`), no SYCL.** Lowest level, no C++ compiler to build. But oneMKL's GPU API requires a `sycl::queue`, so C cannot call oneMKL; it is a separate backend with our own kernels, not a route to oneMKL. Building a `sycl::queue` from Level Zero handles is a DPC++ extension (UNVERIFIED here). Useful later for memory and diagnostics, not as the first path.

**D. Intel's open-source oneMKL Interfaces project (vendor-neutral dispatch).** Possibly relevant to non-Intel SYCL targets; out of scope for Intel hardware. UNVERIFIED, not investigated.

### Recommendation

**A, structured so B is a later swap.** Keep the public Rust types (`Queue`, `UsmBuffer<T>`, `Event`) ours and the shim narrow (a few opaque handles, then blas, then dft), so the runtime half can move to `sycl-rs` once it is stable and tested on Windows without breaking `bonefish` users. Re-evaluate B at R3. C is a later, separate row.

Correction to the HANDOFF plan **[PM]**: it says the shim depends on `onemkl-sys`. The SYCL headers do not use the C headers' types (`oneapi::mkl::transpose` is a C++ enum), so that dependency buys nothing unless we want shared enums. I propose no dependency, to be confirmed in R2.

## 3. Build, distribution, hardware

- A crate whose `build.rs` runs `icpx` cannot build without oneAPI. Recommend a cargo feature `sycl`, off by default (so docs.rs and plain `cargo test` work), plus detection of `ONEAPI_ROOT` with an actionable error. CI that needs oneAPI goes on a **self-hosted runner** (rule 5.1g), never `#[ignore]`.
- Windows: `icpx` targets MSVC, matching Rust's `*-pc-windows-msvc`; runtime DLLs (`sycl9.dll`, `mkl_sycl_*.6.dll`) must be on `PATH` at run time. Not yet exercised; **first measurement row in R2**.
- Linux: Level Zero + oneAPI via `setvars.sh`; no Linux machine inspected (UNVERIFIED).
- GPU: desktop Arc A770 (per HANDOFF; not verified by me). Fuel ROADMAP row 127 is the A770 Vulkan vs oneMKL comparison; oneMKL BLAS has no K-quant kernels, so quantized comparisons are dequantize+GEMM and must be labelled not like-for-like. The "400+ GB" Intel hardware is CireSnave's expectation, unverified here.

## 4. SYCL/DPC++ parser and emitter (assigned to Bonefish by CireSnave 2026-10-30; Unpopped ownership map section 9)

Contract at Unpopped `origin/main`: the emit side is the `Backend` trait (`crates/unpopped/src/backend.rs:621`); the parse side today is `convert.rs`'s tree-sitter `Frontend` (`convert.rs:105`) plus shared vocabulary in `lift.rs`. Unpopped is mid-reshuffle (ownership map; `docs/ir-union-design.md`) and its CUDA/Slang parsers have not yet moved out, so the crate-facing contract is **not stable**. I have not read the trait bodies; R4's design must, first.

Options: **emit first** (SYCL emitter from the IR: elementwise/reduction/scan as `parallel_for`/`nd_range` kernels; testable by compiling with `icpx` and comparing against the `unpopped-cpu-c` reference), then **parse** (SYCL is C++: tree-sitter C++ with a restricted-idiom recogniser, like CUDA's `out[i] = expr;`). Recommend emit first: it is verifiable end to end on the CPU/OpenCL device here, and the parser's idiom set should follow what the emitter produces. The emitter would depend on `unpopped` from crates.io; which version is **[PM]**.

## 5. Proposed rows after this note

R2 `bonefish-sys` shim (queue, USM, event, `gemm` only, then the rest of blas; Windows and Linux builds measured first). R3 safe API + dft. R4a SYCL emitter. R4b SYCL parser. R5 Fuel backend design with the Fuel lane (I do not edit Fuel). R6 onemkl takeover (**[CireSnave]** decision pending).

## 6. Open questions

1. **[PM]** Drop the `onemkl-sys` dependency from the shim plan (section 2)?
2. **[PM]** Approve option A with a swap path to B, and a self-hosted runner with oneAPI (ideally the A770) for GPU-needing tests?
3. **[PM]** Which `unpopped` version/contract for R4a, given the ongoing parser/emitter move?
4. Is the desktop A770 set up with drivers and oneAPI, and may I use it? (Rig access is not mine to assume.)
