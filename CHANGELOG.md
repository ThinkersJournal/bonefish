# Changelog

## 0.0.10

- Docs only: `docs/ASYNC-DESIGN.md`, the design note for asynchronous work (scoped API over an unsafe layer, in-order queues, explicit dependency lists). No code change. Version number is a placeholder for the PM; it stacks after #10's 0.0.9.

## 0.0.9

- **Finding: oneMKL is not host-thread-safe in the shipped build, contrary to the oneAPI specification.** With oneAPI 2026.1 on Windows (CPU/OpenCL device), concurrent first use of `gemm` from four threads crashes inside `mkl_sycl_blas.6.dll` (exception `0xC06D007F` or an access violation): 28 of 40 runs of a standalone C++ program with no bonefish code, and 15 of 15 fresh-process runs of the Rust threading tests. On Linux CI (oneAPI 2026.2, OpenCL CPU, ubuntu runner) the same tests hung. SYCL allocation under threads alone was fine (25 of 25).
- `bonefish-sys`: every oneMKL entry point in the shim (all BLAS, all DFT including plan creation and free) now takes one process-wide mutex. With it, the same Rust threading tests pass 15 of 15 (and 20 of 20 earlier) fresh-process runs, the same binary with `BONEFISH_UNSAFE_NO_MKL_LOCK=1` passes 0 of 15. That switch is a diagnostic only, for retesting a newer oneMKL. Cost: oneMKL calls from different threads take turns; SYCL allocation and copies still overlap.
- `bonefish`: `Queue` and `UsmBuffer<T>` are `Send + Sync` (`Rc` became `Arc`). The basis is the SYCL 2020 rule for SYCL calls plus **our lock** for oneMKL calls, not Intel's thread-safety claim. Unverified: any GPU, other oneMKL releases, Linux with 2026.1.
- `Dft` stays neither `Send` nor `Sync` (the oneAPI specification says class-based objects "may not be shared between concurrent host threads unless otherwise specified"), pinned by `compile_fail,E0277` doctests with `Queue` and `UsmBuffer` as the passing control.
- `bonefish-sys/tests/mkl_lock.rs` fails if a oneMKL call bypasses the lock. The shim exports `bf_mkl_lock_acquisitions` (a counter of entries into the lock path); every foreign function declared in `src/lib.rs` must be classified as oneMKL or not (an unclassified one fails the test, so a new function cannot slip through), each oneMKL function must move the counter by exactly one, and the non-oneMKL ones must leave it unchanged (the control). Checked by mutation: switching `nrm2` from `guard_mkl` to `guard` fails it with "bf_snrm2 moved the oneMKL lock counter by 0, not 1"; reverted.
- Docs: `docs/INTEL-REPORT-DRAFT.md` and the reproducer `docs/intel-report/repro.cpp`, a draft report for Intel with versions, numbers and what was not measured. **Not sent**; whether and where to send it is for CireSnave (board).
- OPEN (not explained, not attributed to load): one `safe_api` test process hung with the lock on (see ROADMAP R8).
- Tests (`tests/threads.rs`): a watchdog turns a hang into a named failure; each operation is run alone (allocation, copies, `gemm`, one DFT plan per thread) with a shared queue, at 1, 2 and 4 threads, plus the control of one queue per thread; the threading tests are serialised against each other. CPU/OpenCL device only; GPU UNVERIFIED.

## 0.0.8

- Test: no tracked file may contain a NUL byte (`tests/no_nul_bytes.rs`; files from `git ls-files`; a tracked binary needs an `ALLOWED_BINARY` entry with a reason). Added after a raw NUL in a Rust byte string compiled, passed every check, and made git treat the file as binary. Negative controls: a planted NUL is found at its offset and the `\0` escape is not flagged.
- ROADMAP R3b: records the SYCL 2020 thread-safety text and why `Queue` stays `!Send + !Sync` (see the row).

## 0.0.7

- `bonefish`: the safe API. `Queue` (`DeviceKind`), `UsmBuffer<T>` (Shared/Device/Host; `f32`, `f64`, `Complex<_>`), `blas::gemm` and `Dft` (complex-to-complex, `f32`/`f64`, 1-D and N-D, in/out-of-place, scales). Every extent is bounds-checked in Rust before the device is touched; a buffer from another queue is `QueueMismatch`. Blocking only, so no `Event` yet. `Queue` is `!Send + !Sync` (thread-safety of `sycl::queue` not verified).
- `bonefish-sys`: without oneAPI the same foreign functions exist as failing stand-ins, so `bonefish` compiles either way and `Queue::new` returns `Error::Unavailable`; `BONEFISH_NO_SYCL=1` builds that way on purpose. Fixed a literal NUL byte in the stand-ins' message.
- CPU/OpenCL device only (Windows laptop, Linux CI); GPU UNVERIFIED.

## 0.0.6

- `bonefish-sys`: BLAS level 1 (`axpy scal copy swap dot nrm2 asum iamax`) and `gemv`, single and double precision, in `shim/blas1.cpp`. Tests are generated from one macro for both precisions and include strided access, `iamax` ties, a transposed `gemv` and a rejected `lda`. CPU/OpenCL device only (Windows laptop, Linux CI); GPU UNVERIFIED. Docs: the `sycl:` proposal now states that its witness proves reachability only, never device admission.

## 0.0.5

- `bonefish-sycl-emit::target`: proposed `sycl:` capability-set vocabulary (36 tokens, each with the DPC++ release that first produces it; `intel_gpu_cri` arrives in 2026.2) with its KISS-CLASSIFY-6.8-0014 derivability witness and a named gate (`tests/sycl_target_tokens.rs`); `docs/sycl-namespace-proposal.md`. ROADMAP R7: icpx fast-math accuracy finding with measured flags. A proposal to KISS, not a registered namespace.

## 0.0.4

- New crate `bonefish-sycl-emit`: SYCL/DPC++ `Backend` for `unpopped` 0.34.11 (pinned exactly). v1 scope: contiguous elementwise, f32/f64, single output; everything else is a typed decline. Math is spelled as `sycl::` calls because icpx's default fast model compiles `expf` to a result 1.8e-4 off. Verified by compiling with icpx and running on the CPU/OpenCL device against `unpopped::oracle`; GPU UNVERIFIED. `publish = false`.

## 0.0.3

- `bonefish-sys`: complex-to-complex DFT (`bf_dft_c2c_new`, `bf_dft_compute`, `bf_dft_free`): single and double precision, 1-D and N-D, in-place and out-of-place, forward/backward scales. Shim split into `shim.cpp`, `dft.cpp`, `common.hpp`. CPU/OpenCL device only; GPU UNVERIFIED. Real-domain transforms not yet covered.

## 0.0.2

- `bonefish-sys`: C shim over oneMKL SYCL (queues, USM, column-major sgemm/dgemm), built with `icpx -fsycl` when oneAPI is found; non-required `oneapi-linux` CI job. Measured on CPU/OpenCL devices only (Windows laptop, Linux CI); GPU UNVERIFIED. Both crates stay `publish = false`.

## 0.0.1

- Docs: R1 survey and design note (`docs/SURVEY.md`); ROADMAP R0 done, R1 in review. No code change; stub crate stays `publish = false`.
