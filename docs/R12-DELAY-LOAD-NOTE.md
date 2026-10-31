# R12 note (INTERNAL, not part of the Intel report): delay-load failure in `mkl_sycl_blas` on Windows

This is the former "finding 4" of `docs/INTEL-REPORT-DRAFT.md`. It is deliberately NOT in the report
that goes to Intel: it has no standalone reproducer (a minimal C++/C probe never crashed). Keep it
here until one exists. ROADMAP R12 is the row.

## delay-load failure inside `mkl_sycl_blas.6.dll` when different BLAS routines first run on parallel threads (Windows; possibly the same cause as finding 1)

## Summary

On Windows, oneAPI 2026.1, CPU (OpenCL) device, a test binary whose 8 tests each call a different set of oneMKL SYCL BLAS routines (complex level 1 and `gemv`, single and double precision) crashed in 11 runs of 100 when the tests ran on parallel threads, each thread with its own queue. Exit code 0xc0000005. Intel's banner "PLEASE submit a bug report" followed by `Exception Code: 0xC06D007F`.

## What was observed (vectored exception handler, first non-C++ exception, Windows)

- `0xC06D007F` (delay-load "procedure not found") raised from `mkl_sycl_blas.6.dll`, naming DLL `mkl_tbb_thread.3.dll` and procedure `mkl_serv_verbose_mode`. Both DLLs are present in the oneAPI `mkl\latestin` directory used.
- Then an access violation **executing** an address that belongs to no loaded module, on another thread, with `sycl9.dll` frames on the stack.
- No test had finished when it happened (the crash is in the first moments, which is why it looks like a cold-start problem).

## Results

| Condition (same test binary) | Crashes / runs |
|---|---|
| default parallel test threads | 11 / 100 (earlier batches 4 to 12 / 100) |
| `RUST_TEST_THREADS=1` | 0 / 100 |
| every routine run once serially before the parallel part | 0 / 100 |
| our wrapper serialising every call into oneMKL (one process-wide lock around submit) | 4 / 100 vs 8 / 100 baseline |
| the same, also waiting for each call to complete inside the lock | 7 / 100 vs 4 / 100 baseline |

## What this does and does not show

- Shown: the crash depends on different routines running cold on parallel threads; serial first use removes it.
- Not shown: the mechanism. Serialising our own calls did not remove it, so it is not simply two host threads inside oneMKL submit at the same time. It may be the same cause as finding 1 (the Windows signature there is also a crash on concurrent first use); that has not been checked.
- **No standalone reproducer.** A minimal C++/C probe (4 and 8 threads, cold, all ten complex routines) did not crash in 40 to 60 runs; only the larger Rust test binary does. Writing one is open.
- Not measured: Linux (the same test passed once on an AMD EPYC runner; one run proves nothing), any GPU, other oneMKL releases.

## Mitigation in use

Bonefish runs one tiny call of each BLAS routine family once, serially, when the first queue of the process is created (`src/warmup.rs`; `BONEFISH_NO_WARMUP=1` turns it off). It is a guard; the mechanism is not shown and ROADMAP R12 stays open.
