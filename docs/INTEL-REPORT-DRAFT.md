# Draft report for Intel: oneMKL SYCL BLAS crashes on concurrent first use from several host threads

This draft holds three findings about the oneMKL SYCL interface, each with its own standalone reproducer: **finding 1** (below) concurrent first use of `gemm` from several host threads crashes; **finding 2** on an in-order queue, `dot` does not wait for the oneMKL call submitted before it; **finding 3** (the last section) committing a DFT descriptor of length 0 kills the process on Intel Xeon CPUs instead of throwing.

Status: **DRAFT, not sent.** Bonefish lane, 2026-10-31. Sending anything outside the portfolio is the PM's and CireSnave's call; this file exists so that call can be made from one page. No secrets, no private paths. Everything below was measured; what was not measured is listed at the end.

## Summary

Calling `oneapi::mkl::blas::column_major::gemm` for the first time from several host threads at once, in one process, crashes intermittently inside Intel's `mkl_sycl_blas` library. It happens whether the threads share one `sycl::queue` or each use their own. The oneAPI specification says "All oneMKL member and non-member functions shall be host thread safe" and "they may be safely called simultaneously from concurrent host threads"; this build does not behave that way. A single mutex around the calls, or one serial warm-up call before the threads start, removes the failures completely in the tests below.

## Environment

| | Windows (where it was characterised) | Linux (CI runner; hang observed, not characterised) |
|---|---|---|
| OS | Windows 11, 10.0.26300.9818 | Ubuntu (GitHub-hosted `ubuntu-latest`) |
| Compiler | Intel oneAPI DPC++/C++ 2026.1.0 (2026.1.0.20260617), target `x86_64-pc-windows-msvc` | oneAPI 2026.2.0-863 from Intel's apt repository |
| oneMKL | install directory `mkl/2026.1`; `mkl_version.h` says `__INTEL_MKL__ 2026`, `MINOR 0`, `UPDATE 1`; library `mkl_sycl_blas.6.dll` | `intel-oneapi-mkl-devel` 2026.2 |
| Device | `sycl::cpu_selector_v`: AMD Ryzen 9 7940HX, OpenCL CPU runtime 2026.21.7.0.24_160000 | AMD EPYC 7763, OpenCL 3.1 (Build 0) 2026.22.10.0.01 |
| GPU | **none involved** | **none involved** |

## Reproducer

`docs/intel-report/repro.cpp` (98 lines, plain C++ plus SYCL and oneMKL, no other dependencies). Build and run lines are in its header. Each configuration is run in a fresh process, because the failure is in first use. Each thread loops 40 times: `malloc_shared` three 2x2 `double` arrays, one `gemm`, `.wait()`, check the result, free.

## Results (Windows, 4 threads x 40 rounds, fresh process per run)

| Configuration | Runs | Clean | Crashed |
|---|---|---|---|
| `own`: each thread makes its own queue, then `gemm` (cold) | 40 | 12 | 28: 18 x exception `0xC06D007F`, 6 x access violation (`0xC0000005`), 4 x other |
| `pre`: queues made serially first, threads only `gemm` | 12 | 10 | 2 x `0xC06D007F` |
| `shared`: one queue for all threads | 12 | 10 | 2 x access violation |
| `own`, 1 thread | 12 | 12 | 0 |
| `own`, 2 threads | 12 | 10 | 1 x `0xC06D007F`, 1 x access violation |
| `own`, `alloc` only (no `gemm`) | 25 | 25 | 0 |
| `shared`, `alloc` only | 25 | 25 | 0 |
| `locked`: `gemm` behind one `std::mutex` | 40 | 40 | 0 |
| `warm1`: one serial `gemm` on the first queue before the threads | 40 | 40 | 0 |
| `warm`: one serial `gemm` on every queue before the threads | 40 | 40 | 0 |

(The failure rate varies with machine load: the `own` row was 8/12 clean on one batch and 4/25 on another before the 12/40 shown. Take the pattern, not one rate.)

What the pattern says: SYCL allocation under threads is fine; the failure needs concurrent oneMKL `gemm`; a mutex or a single serial warm-up call on **one** queue is enough to avoid it, so the race looks like process-wide first-use initialisation inside oneMKL, not per-queue state. That is an inference from the pass/fail pattern, not something observed inside the library.

The same pattern was reproduced through a separate stack (a Rust program calling the same `gemm` through a small C++ shim) with the **same binary**, switching only the lock: 15 of 15 fresh-process runs clean with the lock, 0 of 15 without (11 x `0xC06D007F`, 4 x access violation).

## What the crash looks like

```
PLEASE submit a bug report to https://software.intel.com/en-us/support/priority-support and include the crash backtrace and instructions to reproduce the bug.
Exception Code: 0xC06D007F
 ... mkl_sycl_blas.6.dll ... gemm@column_major@blas@mkl@oneapi ...
```

The symbol names in the backtrace are the nearest *exported* symbol to each address, at large offsets (for example `lookup_and_cache_dpcpp_queue_and_context + 0x8C385`), so they are **not** reliable evidence of which function actually faulted. `0xC06D007F` reads as the Windows delay-load helper's "procedure not found" exception code; that is decoded from the number and was not confirmed against Intel documentation. The other failure mode is a plain access violation.

## Linux

On the CI runner above, a Rust test that runs `gemm` from four threads on one shared queue, and another that runs one DFT plan per thread, each ran for more than 60 seconds (a hang) where the same tests take about a second on Windows. That is a different symptom (a hang, not a crash) on a different OS and oneAPI release, and it has **not** been reduced to the C++ reproducer on Linux. It is mentioned because it is consistent with the same underlying problem, not because it is shown to be.

## What was not measured

- Any GPU. Everything above is the CPU (OpenCL) device.
- The C++ reproducer on Linux, and any oneAPI release other than 2026.1 (Windows) and 2026.2 (Linux CI, Rust tests only).
- Functions other than `gemm` in the C++ reproducer. (The Rust shim now serialises every oneMKL call, so the other functions were not tested unlocked.)
- Whether the fault is in `mkl_sycl_blas`, in the SYCL runtime, or in the OpenCL CPU runtime's interaction with them. The pass/fail pattern points at first-use initialisation; it does not locate the code.
- Whether this is already known or fixed upstream.

## Mitigation in use

Bonefish takes one process-wide mutex around every oneMKL call (`bonefish-sys/shim/common.hpp`, `bf::MklLock`), and offers `BONEFISH_UNSAFE_NO_MKL_LOCK=1` to switch it off so that a newer oneMKL can be retested with the same binary.

# Finding 2: on an in-order queue, oneMKL `dot` does not wait for the call submitted before it

## Summary

On a `sycl::queue` created with `sycl::property::queue::in_order`, two oneMKL calls submitted back to back with no explicit dependency are ordered by the SYCL specification: the second must not start before the first has finished. For most pairs measured that holds. For `oneapi::mkl::blas::column_major::dot` it did not: submitted right after `axpy`, `copy` or `gemm` on the same queue, `dot` read the data from before the earlier call in 98 to 100 of every 100 rounds. When the earlier call's event is passed to `dot` as an explicit dependency, the result was right in every round. Only one thread is involved; this is not the race of finding 1.

## Environment

| | |
|---|---|
| OS | Windows 11, 10.0.26300.9818 |
| Compiler | Intel oneAPI DPC++/C++ 2026.1.0 (2026.1.0.20260617), target `x86_64-pc-windows-msvc` |
| oneMKL | install directory `mkl/2026.1`; `mkl_version.h`: `__INTEL_MKL__ 2026`, `MINOR 0`, `UPDATE 1` |
| Device | `sycl::cpu_selector_v`: AMD Ryzen 9 7940HX (OpenCL CPU runtime) |
| GPU | **none involved** |

## Reproducer

`docs/intel-report/repro_order.cpp` (plain C++ plus SYCL and oneMKL, no other dependencies; build and run lines are in its header). Per round it allocates fresh shared USM, submits two calls back to back without waiting, waits once, and checks whether the second call saw the first call's result. Variant A has no dependency between the calls; variant B passes the first call's event to the second.

## Results (100 rounds per cell, one thread, three runs of the program)

Wrong = the second call used the data from before the first. Right answers: `axpy`->`dot` 70 (stale 10), `copy`->`dot` 30, `gemm`->`dot` 64.

| first call | second call | wrong, no dependency (runs 1, 2, 3) | wrong, explicit dependency |
|---|---|---|---|
| `axpy` | `dot` | 98, 99, 100 | 0, 0, 0 |
| `copy` | `dot` | 100, 100, 100 | 0, 0, 0 |
| `gemm` | `dot` | 100, 100, 100 | 0, 0, 0 |
| `scal` | `axpy` (control) | 0, 0, 0 | 0, 0, 0 |
| `gemm` | `gemm` (control) | 0, 0, 0 | 0, 0, 0 |

The controls show the harness does not simply report every pair as wrong. The same pattern was measured through a separate stack (a Rust program calling the same functions through a small C++ shim): `copy`->`dot` 99 to 100 wrong of 100, `axpy`->`dot` 99 to 100, `gemm`->`dot` 98 to 100 (three runs), and `scal`->`axpy`, `axpy`->`asum`, `axpy`->`nrm2`, `gemm`->`gemm`, `dot`->`axpy` all 0.

## What this does and does not show

- It shows a pair of calls that the SYCL in-order rule says are ordered, run out of order, with the explicit-dependency form correct. It does not show **why**: one reading is that `dot` (which writes a scalar through a USM pointer) is executed or enqueued outside the queue's order, but that is a guess from the pass/fail pattern, not something observed inside the library.
- Only `axpy`, `copy`, `gemm`, `scal`, `asum`, `nrm2` and `dot` were exercised as the "first" or "second" call, and only some pairs. Other calls (`iamax`, level 2, the DFT interface) were not measured for this ordering effect.
- Not measured: any GPU; the Linux build (oneAPI 2026.2) with this reproducer; any oneMKL other than 2026.1.

## Mitigation in use

Bonefish chains every asynchronous submission to the queue's previous one (each call lists the queue's last event as a dependency; `bonefish-sys/shim/common.hpp`, `deps_chained`), so it does not rely on the queue's order for oneMKL. `BONEFISH_UNSAFE_NO_ASYNC_CHAIN=1` turns the chain off so the same binary can show whether a newer oneMKL still needs it.

# Finding 3: committing a DFT descriptor of length 0 crashes with SIGFPE on Intel Xeon CPUs

## Summary

`oneapi::mkl::dft::descriptor<..., COMPLEX>` of length 0, followed by `commit(queue)` on a CPU queue, is refused with the exception `FFT_INVALID_DESCRIPTOR` on AMD EPYC, which is the behaviour an invalid descriptor should have. On Intel Xeon CPUs with default instruction-set dispatch the process is instead killed by `SIGFPE` (exit 136 on Linux, "core dumped") after Intel's "PLEASE submit a bug report ... include the crash backtrace" banner. The same call on the same Intel hosts with `MKL_ENABLE_INSTRUCTIONS=AVX2` throws `FFT_INVALID_DESCRIPTOR`. Valid lengths work everywhere measured. One thread; no queue sharing; not a race.

## Environment

| | |
|---|---|
| OS | GitHub-hosted runners, image `ubuntu-24.04` (Linux) |
| Compiler | Intel oneAPI DPC++/C++ 2026.2.0 (2026.2.0.20261001), from Intel's apt repository |
| oneMKL | `mkl_version.h`: `__INTEL_MKL__ 2026`, `MINOR 0`, `UPDATE 2` |
| OpenCL CPU runtime | 2026.22.10.0.01 (as printed by `sycl-ls`) |
| Device | `sycl::cpu_selector_v` |
| GPU | **none involved** |
| CPUs that crashed | Intel Xeon Platinum 8370C @ 2.80 GHz; Intel Xeon Platinum 8573C |
| CPUs that did not | AMD EPYC 7763, 9V45, 9V74; also a Windows laptop, AMD Ryzen 9 7940HX, oneAPI 2026.1 (threw) |

The runners were assigned by the CI service, not chosen; the CPU of each was read from `lscpu` and `sycl-ls` in the same job.

## Reproducer

`docs/intel-report/repro_dft.cpp` (plain C++ plus SYCL and oneMKL, no other dependencies; build and run lines are in its header). Usage `repro_dft <s|d> <inplace 0|1> <n> [rounds]`: builds a 1-D complex-to-complex descriptor of length `n`, commits it on a CPU queue, transforms a delta and checks the answer. It prints a "stage:" line before each step, so the last one printed places a crash.

## Results

| Measurement | Intel Xeon runners | AMD EPYC runners |
|---|---|---|
| `n = 0`, both precisions, in-place and not (4 configurations per runner), default dispatch | 3 runners x 4 = 12 of 12 killed by SIGFPE (exit 136) | 13 runners x 4 = 52 of 52 threw `FFT_INVALID_DESCRIPTOR`; no crash |
| the same with `MKL_ENABLE_INSTRUCTIONS=AVX2` | 12 of 12 threw `FFT_INVALID_DESCRIPTOR` | 52 of 52 threw `FFT_INVALID_DESCRIPTOR` (same as default) |
| `n = -1` | threw `FFT_ALLOC_FAILED` (24 of 24: 12 per dispatch setting) | threw `FFT_ALLOC_FAILED` (104 of 104) |
| where it dies (stage markers, one Intel runner, Xeon 8370C) | last stage printed is "committing": the crash is inside `commit` | threw `FFT_INVALID_DESCRIPTOR` from `commit` (15 of 15 AMD runners) |
| valid lengths 2, 4, 5, 8, 12, 16, 30, 64, 100 x single/double x in-place/out-of-place (36 configurations) | all correct on 3 Intel Xeon runners (one 8573C, two 8370C) | all correct on 13 AMD runners |

The same crash was first found through a Rust test that creates a descriptor of length 0 and expects an error: that test binary died with SIGFPE in 30 of 30 runs on a Xeon 8370C with default dispatch and passed 30 of 30 with `MKL_ENABLE_INSTRUCTIONS=AVX2`; on each of 9 AMD runners it passed 30 of 30 in both settings. Run alone, 3 of 3 on each of three Intel runners (two Xeon 8573C, one 8370C) died with default dispatch and passed with AVX2; the other six tests in that file passed everywhere.

## What this does and does not show

- It shows that, on these Xeons, oneMKL turns an invalid descriptor into a process kill inside `commit`, while the same build refuses it with an exception on AMD and with AVX2 forced. The AVX2 result suggests the default dispatch picks a code path that divides by the length, but **which instruction-set path default dispatch selects on these CPUs was not measured**, and the other `MKL_ENABLE_INSTRUCTIONS` values (AVX512 and the AVX512_E* levels) were not tried.
- Only a length of 0 was found to crash. Other invalid descriptors (other configuration parameters, strides, multi-dimensional shapes with a zero) were not tried in the standalone program; `[4, 0]` and `[0, 4]` were refused by Bonefish before reaching oneMKL and so are not evidence either way.
- Not measured: Windows on Intel hardware; any GPU; Intel client CPUs; other oneMKL releases; the C `DFTI` interface (the report is about the SYCL interface only).

## Mitigation in use

Bonefish refuses a DFT length below 1 itself, before it reaches oneMKL (`bonefish-sys/shim/dft.cpp`), and its test checks that the refusal is its own, not oneMKL's.
