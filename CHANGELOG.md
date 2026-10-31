# Changelog

## 0.0.14

- **Fix: `bf_dft_c2c_new` refuses a DFT length below 1 itself, before oneMKL.** Committing a oneMKL SYCL DFT descriptor of length 0 kills the process with SIGFPE on Intel Xeon CPUs under default instruction-set dispatch (Xeon Platinum 8370C and 8573C, oneAPI 2026.2, Linux CI), while AMD EPYC and `MKL_ENABLE_INSTRUCTIONS=AVX2` get the expected `FFT_INVALID_DESCRIPTOR` exception. This is what made the Linux `oneapi` CI job fail intermittently on `bonefish-sys --test dft` (main@772c5bc post-merge, #15 twice): the runner CPU is assigned at random and every failure was on a Xeon, every pass on an AMD EPYC. It is not caused by the async work: the commit before it (7ecf94a) crashes the same way. Cause pinned by a standalone C++ program with no Bonefish code, `docs/intel-report/repro_dft.cpp` (12 of 12 crashes on 3 Intel runners, 0 on AMD, stage markers place it inside `commit`); written up as finding 3 in `docs/INTEL-REPORT-DRAFT.md` (draft, **not sent**). The safe `Dft::c2c` already refused 0; only the raw shim entry (and the sys-level test that calls it) passed it through.
- Test: `invalid_plan_is_an_error_not_a_crash` now checks five bad shapes (`[0]`, `[4, 0]`, `[0, 4]`, `[-1]`, `[4, -3]`) and that the refusal carries the shim's own message, so it cannot pass by oneMKL happening to refuse; with a valid length as the control. Mutation: with the guard disabled the test fails here with 'refused by oneMKL or not at all, not by the shim: FFT_INVALID_DESCRIPTOR'.
- ROADMAP R9 records the measurements and what was NOT measured (default dispatch path, AVX512 levels, other invalid descriptors, Windows on Intel hardware, DFTI, any GPU). Version number is a placeholder for the PM.

## 0.0.13

- `bonefish`: scoped non-blocking level 1 (`axpy`, `scal`, `copy`, `swap`, `dot`, `nrm2`, `asum`, `iamax`), `gemv` and DFT (`Scope::plan` binds a `Dft` with a shared borrow; `dft_forward`/`dft_backward`, in place and out of place) on `Scope`, for `f32` and `f64` (R3c step 3). Vector increments must be at least 1. Refusals before anything is submitted: a buffer too short for its extent, a zero increment, a result buffer that is empty, an output that is the same buffer as an input, a plan of the wrong placement, a plan or buffer from another queue. `i64` is now an `Element` (the index `iamax` writes).
- `bonefish-sys`: `bf_{s,d}{axpy,scal,copy,swap,dot,nrm2,asum,iamax,gemv}_async` and `bf_dft_compute_async` (each returns the event, takes a dependency array; only the submission is under the oneMKL lock), and the DFT plan handle now carries its queue's error store so an async DFT's errors are attributed to that queue.
- **Retraction (reaches `docs/ASYNC-DESIGN.md` 3.4/6 and the `Queue::scope` docs, replaced in this release): an in-order queue does not order oneMKL calls.** 0.0.11 and 0.0.12 said the scope's operations cannot overlap because the queue is in-order (a "soundness requirement" the SYCL specification supplies). That was never measured for oneMKL and is false on this build: on an `in_order` queue, `dot` submitted right after `axpy`/`copy`/`gemm` read the old data in 98 to 100 of 100 rounds (Windows, oneAPI 2026.1, CPU/OpenCL; standalone C++, three runs); 0 of 100 with the earlier event passed as a dependency; `scal`->`axpy`, `axpy`->`asum`/`nrm2`, `gemm`->`gemm`, `dot`->`axpy` were ordered. The only operation the 0.0.12 safe API exposed was `gemm`, and `gemm`->`gemm` was ordered, so no wrong result from 0.0.12 is known; the argument was wrong, and it would have broken with the next operation added.
- Fix: the shim chains every asynchronous submission to the queue's previous asynchronous event (`deps_chained`), so the order does not depend on which calls the library happens to order. With the chain on all eight measured pairs are right in every round; with it off the same binary reproduces the stale reads. `BONEFISH_UNSAFE_NO_ASYNC_CHAIN=1` (or `bf_test_set_async_chain(0)`) turns the chain off: a diagnostic for exactly that A/B and for retesting a newer oneMKL, not a feature. Safe-API end-to-end test `dot_straight_after_axpy_sees_the_axpy`: passes 5 of 5 with the chain, fails 5 of 5 without (10 instead of 70).
- Docs: `docs/INTEL-REPORT-DRAFT.md` finding 2 with the standalone reproducer `docs/intel-report/repro_order.cpp` (draft, **not sent**). `bonefish-sys/tests/mkl_lock.rs` classifies and counts the 19 new functions (mutation: submitting without the lock fails with "bf_saxpy_async moved the oneMKL lock counter by 0, not 1").
- Not measured: any GPU; Linux locally; other oneMKL releases; `iamax` and the DFT call without the chain; a failing kernel (as opposed to the forced host-task error) in the async level-1/DFT paths.
- Version number is a placeholder for the PM (stacks after 0.0.12).

## 0.0.12

- `bonefish`: the scoped non-blocking API (R3c step 2, `docs/ASYNC-DESIGN.md`). `Queue::scope(|s| ...)` gives a `Scope` on which `gemm`/`gemm_after` (f32, f64) return an `Event<'scope>` without waiting; buffers are bound once with `s.read(&buf)` / `s.write(&mut buf)` (a `Read`/`Write` handle holding a shared/exclusive borrow until the scope ends), and the scope waits for the queue on normal return, on error and on panic, then returns the first asynchronous error. Bounds are checked as in the blocking `gemm`; an output that is also an input of the same call is `InvalidArgument`; a buffer from another queue is `QueueMismatch` at binding. `Scope` is `Send` and `!Sync`. The blocking functions are unchanged.
- Pinned by tests: `compile_fail,E0502` (host read while bound for writing), `compile_fail,E0505` (drop while bound for reading), a compile_fail that an `Event` cannot leave the scope (the compiler's error is "lifetime may not live long enough" at the returned value), `compile_fail,E0277` for `Scope: Sync` with a `Send` control; `tests/scope.rs` (dependent chain, explicit event dependency, queue mismatch, aliasing, bounds, **a panic inside the scope still waits**: mutation-checked, with the wait disarmed that test fails 12 of 12 runs, and passes with it); an in-crate test that an asynchronous failure comes back from `scope` and is not reported twice.
- **Fix to 0.0.11 (shim): asynchronous errors are now kept per queue, not per thread.** 0.0.11 collected them in a per-thread list on the assumption that SYCL runs the async handler on the waiting thread; it does not promise that. Measured on Windows (oneAPI 2026.1, CPU) with four threads on four queues: 138 of 300 failures lost on failing queues (343 of 600 in another shape) and a clean queue once reported another queue's error. Now `bf_queue_new` gives each queue an error store its handler captures, event handles carry their queue's store, and each wait drains its own queue's. After the fix: 0 foreign errors in 9 runs, 0 of 150 lost single-threaded. Remaining, and not ours: with four threads active about 2 to 5 in 100 errors are not yet posted when the first wait returns; all 66 of 66 missed ones were reported by a later wait on the same queue. So `Ok` from `scope` is not proof the work succeeded; documented on `Queue::scope`. A failing kernel (as opposed to the forced host-task error) was not measured; GPU UNVERIFIED. Found when the Linux oneapi CI job failed once on the in-crate scope test (lost error, then passed on rerun).
- Not shown: that in-order matters for correctness on any device (the CPU device ran an out-of-order queue in order, ROADMAP R3c); any GPU. Level 1/`gemv` and `Dft` are still blocking-only.
- Version number is a placeholder for the PM (stacks after 0.0.11).

## 0.0.11

- `bonefish-sys`: the non-blocking shim layer (R3c step 1, `docs/ASYNC-DESIGN.md`). Every queue is now in-order and has an `async_handler`; new `bf_sgemm_async`/`bf_dgemm_async` (return a `sycl::event*`, take an array of dependency events; only the submission is under the oneMKL lock), `bf_event_wait`, `bf_event_free`, `bf_queue_wait`, `bf_queue_is_in_order`, and a diagnostic `bf_test_async_failure`. Asynchronous errors come back as status 1 through both wait calls. No safe-API change yet (the scoped API is step 2), so no `bonefish` behaviour changes except that its queues are in-order.
- Measured, not assumed: without the `async_handler` an asynchronous failure kills the process (the test fails by exit code 127, mutation-checked); with `in_order` removed the dependent-chain test still passes on the CPU device (15/15), so it has no teeth there and `is_in_order()` is what is pinned. GPU UNVERIFIED.
- `bonefish-sys/tests/mkl_lock.rs` classifies and counts the new functions. ROADMAP: R3c added; R8 gains a second unexplained stall (first run of the new test binary, not reproduced in 40 runs); the tests now carry a 60 s watchdog.
- Version number is a placeholder for the PM (stacks after 0.0.10).

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
