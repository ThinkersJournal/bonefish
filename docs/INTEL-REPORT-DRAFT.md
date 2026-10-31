# Draft report for Intel: oneMKL SYCL BLAS crashes on concurrent first use from several host threads

This draft holds two findings about the oneMKL SYCL interface, each with its own standalone reproducer: **finding 1** (below) concurrent first use of `gemm` from several host threads crashes; **finding 2** (the last section) on an in-order queue, `dot` does not wait for the oneMKL call submitted before it.

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
