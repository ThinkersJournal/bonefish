# oneMKL SYCL interface: three issues, each with a standalone reproducer

Status: **ready to send; not sent.** The sender's name and contact line go here when it is sent.
Everything below was measured; what was not measured is stated in each section. No GPU was
involved in any measurement: every number is the CPU (OpenCL) device.

## One-page summary

We use the oneMKL SYCL interface (`oneapi::mkl::blas::column_major`, `oneapi::mkl::dft`) from a
wrapper library and found three behaviours that differ from the specification or from what an
invalid argument should do. Each has a single-file C++ reproducer (no other dependencies) in
`docs/intel-report/`. Build and run:

```
icpx -fsycl -O2 -qmkl=parallel repro.cpp -o repro               # finding 1; run: repro own 4 40
icpx -fsycl -O2 -qmkl=parallel repro_order.cpp -o repro_order   # finding 2; run: repro_order 100
icpx -fsycl -O2 -qmkl=parallel repro_dft.cpp -o repro_dft       # finding 3; run: repro_dft d 0 0
```

`docs/intel-report/run_all.sh` builds all three and prints every table in this report (Linux, or
Windows in Git Bash from a oneAPI command prompt).

| # | What we see | Where it reproduced | What we ask Intel |
|---|---|---|---|
| 1 | `gemm` called for the first time from several host threads at once crashes inside `mkl_sycl_blas` | **Reproduced on one machine (Windows, oneAPI 2026.1, 28/40 crashes); not reproduced on Linux/EPYC oneAPI 2026.2 (0/400).** Across 18 runner jobs in all (7 Linux, 6 Windows Server with 2026.2, 5 Windows Server with 2026.1, the same compiler build as the machine that crashed): 7197 of 7200 runs clean; the other 3 runs, all in one cell of one 2026.1 job, exited 127 (shell "command not found", not a crash code) and are **unexplained**; they did not recur in the 3 later 2026.1 jobs | Is first use from several threads meant to be safe (the specification says all oneMKL functions are host thread safe)? If so, please look at the crash; we can supply more logs. |
| 2 | On an `in_order` queue, `dot` submitted right after `axpy`, `copy` or `gemm` does not wait for it: stale data in 97 to 100 of every 100 rounds. With the earlier call's event passed as a dependency it is correct. | **Reproduced on every machine tried**: Windows 11 laptop, Ubuntu 24.04 runners, Windows Server 2025 runners; oneAPI 2026.1 and 2026.2 | Please confirm whether `dot` should respect the in-order queue, and fix or document it. |
| 3 | Committing a DFT descriptor of length 0 kills the process with SIGFPE on Intel Xeon CPUs (default dispatch) instead of throwing | **Reproduced on every Intel Xeon runner tried** (3 of 3); AMD EPYC runners, Windows Server on EPYC and the Windows 11 laptop throw `FFT_INVALID_DESCRIPTOR`; so do the Xeons with `MKL_ENABLE_INSTRUCTIONS=AVX2` | Please make an invalid length throw on every dispatch path. |

How firm each one is: findings 2 and 3 reproduce on every machine where they were tried and have
controls that pass. Finding 1 is a race that we have caught on a single machine; the measurements
on other machines are nulls apart from 3 unexplained exit-127 runs, and a null there does not show
the crash is absent elsewhere. Its
failure rate on that machine also varied from 28 in 40 down to 0 in 40 with machine load, so a
short run there can pass.

## Machines

All CPU (OpenCL) device runs. "Runner" means a GitHub-hosted runner, whose CPU is assigned by the
service and was read from the same job (`lscpu` or `Get-CimInstance`, and `sycl-ls`).

| id | OS | CPU | DPC++ compiler | oneMKL (`mkl_version.h`) | OpenCL CPU runtime |
|---|---|---|---|---|---|
| M1 | Windows 11, 10.0.26300.9818 (laptop) | AMD Ryzen 9 7940HX, 32 logical CPUs | 2026.1.0 (2026.1.0.20260617) | 2026, minor 0, update 1 | 2026.21.7.0.24_160000 |
| M2 | Ubuntu 24.04.5, kernel 7.0.0-1017-azure (runner, 4 vCPUs) | AMD EPYC 9V74 (4 jobs), AMD EPYC 7763 (3 jobs) | 2026.2.0 (2026.2.0.20261001), Intel apt repository | 2026, minor 0, update 2 | 2026.22.10.0.01_160000 |
| M3 | Windows Server 2025 Datacenter, 10.0.26100 (runner) | AMD EPYC 9V45 (3 jobs), 9V74 (2 jobs), 7763 (1 job) | 2026.2.0 (2026.2.0.20261001), toolkit 2026.2.0.661 | 2026, minor 0, update 2 | 2026.22.10.0.01_160000 |
| M4 | Windows Server 2025 Datacenter, 10.0.26100 (runner) | AMD EPYC 7763 (4 jobs), 9V45 (1 job) | 2026.1.0 (2026.1.0.20260617), toolkit 2026.1.0.191 | 2026, minor 0, update 1 | 2026.21.6.0.17_160000 |
| M5 | Ubuntu 24.04 (runner) | Intel Xeon Platinum 8370C (2 runners), 8573C (1 runner) | 2026.2.0 (2026.2.0.20261001) | 2026, minor 0, update 2 | 2026.22.10.0.01_160000 |

M4 has the same compiler build as M1. M3 and M4 were installed from Intel's offline toolkit
installer (components: DPC++ compiler and oneMKL development files). M2 and M5 used the apt
packages `intel-oneapi-compiler-dpcpp-cpp`, `intel-oneapi-mkl-devel` and
`intel-oneapi-runtime-opencl`. Other AMD runners (EPYC 7763, 9V45, 9V74) also appear among the 13
in finding 3.

---

# Finding 1: concurrent first use of `gemm` from several host threads crashes (one machine)

## Summary

Calling `oneapi::mkl::blas::column_major::gemm` for the first time from several host threads at
once, in one process, crashed intermittently inside `mkl_sycl_blas` on M1. It happened whether the
threads shared one `sycl::queue` or each used their own. The oneAPI specification says "All oneMKL
member and non-member functions shall be host thread safe" and "they may be safely called
simultaneously from concurrent host threads"; M1 does not behave that way. A single mutex around
the calls, or one serial warm-up call before the threads start, removed the failures in the tests
below.

## Reproducer

`repro.cpp` (plain C++ plus SYCL and oneMKL). Each configuration is run in a **fresh process**,
because the failure is in first use. Each thread loops 40 times: `malloc_shared` three 2x2
`double` arrays, one `gemm`, `.wait()`, check the result, free. Arguments:
`repro <mode> <threads> <rounds> [alloc]`. Exit code 0 clean, 3 wrong result, 42 watchdog (a hang),
anything else a crash.

## Results on M1 (4 threads x 40 rounds, fresh process per run)

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

The failure rate on M1 varies with machine load: the `own` row was 8 of 12 clean on one batch and
4 of 25 on another before the 12 of 40 shown. A later re-run of the same reproducer source on a
quiet M1 (`own`, 4 threads, 40 runs): 40 of 40 clean, then 39 of 40 clean; with the process
restricted to 4 logical CPUs (affinity mask) 37 of 40 and 23 of 40 clean. So the crash is real on
M1 but its rate ranges from about 0 to 70 percent; take the pattern, not one rate.

The pass/fail pattern says: SYCL allocation under threads is fine; the failure needs concurrent
oneMKL `gemm`; a mutex or one serial warm-up call on **one** queue avoids it, so the race looks
like process-wide first-use initialisation inside oneMKL rather than per-queue state. That is an
inference, not something observed inside the library.

The same pattern was seen through a separate stack on M1 (a Rust program calling the same `gemm`
through a small C++ shim) with the same binary, switching only the lock: 15 of 15 fresh-process
runs clean with the lock, 0 of 15 without (11 x `0xC06D007F`, 4 x access violation).

## Results on the other machines (same reproducer source, `run_all.sh 40`)

Every cell: 40 fresh-process runs, 4 threads x 40 rounds (`own, 1 thread` and `own, 2 threads` as
named).

| Machine (jobs) | `own` | `pre` | `shared` | `own` 1 thr | `own` 2 thr | alloc only (both) | `locked` | `warm1` | `warm` |
|---|---|---|---|---|---|---|---|---|---|
| M2 Ubuntu / EPYC, 2026.2 (7 jobs) | 40/40 clean in every job | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| M3 Windows Server, 2026.2 (6 jobs) | 40/40 in every job | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| M4 Windows Server, 2026.1 (5 jobs) | 40/40 in 4 jobs; **37/40 in one job (see below)** | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |

Totals over all 18 jobs (10 cells x 40 runs each): 7200 runs, 7197 clean, 3 not clean. M2: 2800 of
2800 clean. M3: 2400 of 2400. M4: 1997 of 2000. **The 3 non-clean runs are unexplained.** They are
3 of the 40 runs of `own` (the first cell of the script, so run soon after the executable was
built; which of the 40 runs they were was not recorded) in one M4 job (CPU: EPYC 9V45); each exited 127, which a POSIX shell uses for "command not found" and
is not a Windows crash code, and the run's output had been discarded, so we do not know whether it
was the executable failing to start or the crash of finding 1. The script now prints the output of
the first non-clean run in every cell; the six jobs run after that
change (two each on M2, M3 and M4) had no non-clean run, so there is still no captured output of it. Treat it as a possible
sighting of finding 1 on a runner, not as one.

Apart from those 3, **the crash was not reproduced on any runner**, including M4, which has the
same compiler build as M1 and runs Windows. What differs from M1 (CPU model and core count,
Windows 11 versus Windows Server, load, the runner's virtualisation) was not isolated; restricting
M1 to 4 logical CPUs did not make the crash disappear (it was in fact more frequent), so core count
alone does not explain the difference.

## What the crash looks like on M1

```
PLEASE submit a bug report to https://software.intel.com/en-us/support/priority-support and include the crash backtrace and instructions to reproduce the bug.
Exception Code: 0xC06D007F
 ... mkl_sycl_blas.6.dll ... gemm@column_major@blas@mkl@oneapi ...
```

The symbol names in the backtrace are the nearest *exported* symbol at large offsets, so they are
not reliable evidence of which function faulted. `0xC06D007F` is the Windows delay-load helper's
"procedure not found" exception code (decoded from the number; not confirmed against Intel
documentation). The other failure mode is a plain access violation.

## What was not measured

- Any GPU; any Intel CPU for this finding; Windows 11 other than M1; any other machine with M1's
  core count.
- The cause: whether the fault is in `mkl_sycl_blas`, the SYCL runtime, or the OpenCL CPU runtime's
  interaction with them. Whether it is already known or fixed.
- Functions other than `gemm` in the standalone reproducer.

## Mitigation we use

One process-wide mutex around every oneMKL call, and a serial first call of each routine before
any other thread starts.

---

# Finding 2: on an in-order queue, oneMKL `dot` does not wait for the call submitted before it

## Summary

On a `sycl::queue` created with `sycl::property::queue::in_order`, two oneMKL calls submitted back
to back with no explicit dependency are ordered by the SYCL specification: the second must not
start before the first has finished. For most pairs measured that holds. For
`oneapi::mkl::blas::column_major::dot` it did not: submitted right after `axpy`, `copy` or `gemm`
on the same queue, `dot` read the data from before the earlier call in 97 to 100 of every 100
rounds on every machine tried. When the earlier call's event is passed to `dot` as an explicit
dependency, the result was right in every round. One thread; this is not finding 1's race.

## Reproducer

`repro_order.cpp`. Per round it allocates fresh shared USM, submits two calls back to back without
waiting, waits once, and checks whether the second call saw the first call's result. Variant A has
no dependency between the calls; variant B passes the first call's event to the second. Right
answers: `axpy`->`dot` 70 (stale 10), `copy`->`dot` 30, `gemm`->`dot` 64.

## Results (wrong rounds out of 100; three runs of the program per job, min to max)

| first call -> second | M1 | M2 (7 jobs, 21 runs) | M3 (6 jobs, 18 runs) | M4 (5 jobs, 15 runs) | with explicit dependency (all machines) |
|---|---|---|---|---|---|
| `axpy` -> `dot` | 98 to 100 | 98 to 100 | 98 to 100 | 97 to 100 | 0 |
| `copy` -> `dot` | 100 | 99 to 100 | 98 to 100 | 98 to 100 | 0 |
| `gemm` -> `dot` | 100 | 99 to 100 | 98 to 100 | 98 to 100 | 0 |
| `scal` -> `axpy` (control) | 0 | 0 | 0 | 0 | 0 |
| `gemm` -> `gemm` (control) | 0 | 0 | 0 | 0 | 0 |

The controls show the harness does not simply report every pair as wrong. The same pattern was
measured on M1 through a separate stack (a Rust program calling the same functions through a small
C++ shim): `copy`->`dot` 99 to 100 wrong of 100, `axpy`->`dot` 99 to 100, `gemm`->`dot` 98 to 100,
and `scal`->`axpy`, `axpy`->`asum`, `axpy`->`nrm2`, `gemm`->`gemm`, `dot`->`axpy` all 0.

## What this does and does not show

- It shows a pair of calls that the SYCL in-order rule says are ordered, run out of order, with the
  explicit-dependency form correct, on two operating systems, two oneAPI releases and three AMD CPU
  models. It does not show **why**: one reading is that `dot` (which writes a scalar through a USM
  pointer) is executed or enqueued outside the queue's order, but that is a guess from the pass/fail
  pattern.
- Only `axpy`, `copy`, `gemm`, `scal`, `asum`, `nrm2` and `dot` were exercised as the first or
  second call, and only some pairs. `iamax`, level 2 and the DFT interface were not measured for
  this effect.
- Not measured: any GPU; Intel CPUs for this finding; other oneMKL releases.

## Mitigation we use

Every asynchronous submission lists the queue's previous event as an explicit dependency, so we do
not rely on the queue's order for oneMKL.

---

# Finding 3: committing a DFT descriptor of length 0 crashes with SIGFPE on Intel Xeon CPUs

## Summary

`oneapi::mkl::dft::descriptor<..., COMPLEX>` of length 0, followed by `commit(queue)` on a CPU
queue, is refused with the exception `FFT_INVALID_DESCRIPTOR` on AMD EPYC, which is the behaviour
an invalid descriptor should have. On Intel Xeon CPUs with default instruction-set dispatch the
process is instead killed by `SIGFPE` (exit 136 on Linux, "core dumped") after Intel's "PLEASE
submit a bug report ... include the crash backtrace" banner. The same call on the same Intel hosts
with `MKL_ENABLE_INSTRUCTIONS=AVX2` throws `FFT_INVALID_DESCRIPTOR`. Valid lengths work everywhere
measured. One thread; no queue sharing; not a race.

## Reproducer

`repro_dft.cpp`. Usage `repro_dft <s|d> <inplace 0|1> <n> [rounds]`: builds a 1-D
complex-to-complex descriptor of length `n`, commits it on a CPU queue, transforms a delta and
checks the answer. It prints a "stage:" line before each step, so the last one printed places a
crash. Exit 0 ok, 3 wrong, 4 threw; a crash is the finding.

## Results

| Measurement | Intel Xeon runners (M5) | AMD EPYC runners, Linux (13 runners, M2 among them) |
|---|---|---|
| `n = 0`, both precisions, in place and not (4 configurations per runner), default dispatch | 3 runners x 4 = 12 of 12 killed by SIGFPE (exit 136) | 52 of 52 threw `FFT_INVALID_DESCRIPTOR`; no crash |
| the same with `MKL_ENABLE_INSTRUCTIONS=AVX2` | 12 of 12 threw `FFT_INVALID_DESCRIPTOR` | 52 of 52 threw (same as default) |
| `n = -1` | threw `FFT_ALLOC_FAILED` (24 of 24: 12 per dispatch setting) | threw `FFT_ALLOC_FAILED` (104 of 104) |
| where it dies (stage markers, one runner, Xeon 8370C) | last stage printed is "committing": the crash is inside `commit` | threw from `commit` (15 of 15 runners) |
| valid lengths 2, 4, 5, 8, 12, 16, 30, 64, 100 x single/double x in-place/out-of-place (36 configurations) | all correct on 3 Intel Xeon runners | all correct on 13 AMD runners |

Also measured with the same reproducer, `n = 0`: M1 (Windows 11, oneAPI 2026.1) threw; so did all
18 runner jobs of finding 1 (7 on M2, 6 on M3, 5 on M4; 4 configurations each, default dispatch and
AVX2: 144 of 144 threw). **Windows on an Intel CPU was not measured.**

The same crash was first found through a Rust test that creates a descriptor of length 0 and
expects an error: that test binary died with SIGFPE in 30 of 30 runs on a Xeon 8370C with default
dispatch and passed 30 of 30 with `MKL_ENABLE_INSTRUCTIONS=AVX2`; on each of 9 AMD runners it
passed 30 of 30 in both settings.

## What this does and does not show

- On these Xeons oneMKL turns an invalid descriptor into a process kill inside `commit`, while the
  same build refuses it with an exception on AMD and with AVX2 forced. The AVX2 result suggests the
  default dispatch picks a path that divides by the length, but **which instruction-set path default
  dispatch selects on these CPUs was not measured**, and the other `MKL_ENABLE_INSTRUCTIONS` values
  (AVX512 and the AVX512_E* levels) were not tried.
- Only a length of 0 was found to crash. Other invalid descriptors (other configuration parameters,
  strides, multi-dimensional shapes with a zero) were not tried in the standalone program.
- Not measured: Windows on Intel hardware; any GPU; Intel client CPUs; other oneMKL releases; the C
  `DFTI` interface (this report is about the SYCL interface only).

## Mitigation we use

We refuse a DFT length below 1 ourselves, before it reaches oneMKL.
