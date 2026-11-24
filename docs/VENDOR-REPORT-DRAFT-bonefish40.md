# DRAFT, NOT SENT: loader-lock deadlock between Intel OpenCL's `tbb12` DllMain and NVIDIA's `nvdiagclt64` (Windows)

Status: internal draft, not sent. Strip this header before sending. Evidence is from one laptop; no other machine has run it.

## Summary

Several threads of one process each creating a SYCL queue for the first time hang with zero CPU.
Thread A is inside `LoadLibrary` of Intel's OpenCL runtime; `tbb12.dll`'s `DllMain` (loader lock held)
calls `WinVerifyTrust`, which waits on CRYPT32's process-wide default chain engine lock. Thread B is an
NVIDIA background thread (`nvdiagclt64`) that holds that lock while it creates the engine and then needs
a delay-loaded DLL, which waits on the loader lock held by thread A. Neither can proceed.

## Environment (read on the machine, 2026-11-23)

- Windows 11 Pro 10.0.26300
- GPU NVIDIA GeForce RTX 4070 Laptop, driver 621.24 (`nvidia-smi`); `nvopencl64.dll` file version 32.0.16.2124 (DriverStore `nvamsi.inf_amd64_b8df733b9c1759fb`)
- Intel oneAPI 2026.1: `intelocl64.dll` 2026.7.0.0.0724 (signature Valid), `sycl9.dll` 2026.1.0.0 (Valid), `tbb12.dll` from `oneAPI\tbb\latest\bin`, file version 2023.1.0 (Valid)
- Rust test binary calling a C++ shim (`sycl::queue` constructor), debug build

## Reproduction

`bonefish-sys` `tests/level3.rs` has 8 tests; the default harness runs them on 8 threads and each makes
its own queue (`sycl::queue(sycl::cpu_selector_v, ...)`).

| run | result |
|---|---|
| full workspace `cargo test`, default threads, watchdog 900 s | hung in `level3`, watchdog fired |
| `level3.exe` alone, sequential, 25 runs | 25 pass |
| `level3.exe` x 4 processes at once, loop, no fix | all 4 stalled (loop runs 2, 3, 15, 15) |
| same with a process-wide mutex around `sycl::queue` construction | stalled (runs 9, 32, 32, 32) |
| same with `ONEAPI_DEVICE_SELECTOR=opencl:cpu` | stalled (runs 1, 1, 7, 34) |
| same with `OCL_ICD_FILENAMES=<intelocl64.dll only>` | stalled (runs 1, 1, 4, 4); `nvdiagclt64` still loaded |
| same with one `WinVerifyTrust(tbb12.dll)` call before the first queue (our workaround) | 4 x 120 runs, no stall |
| full workspace, default threads, with the workaround | 3 of 3 green |

## Stacks (noninvasive `cdb -pv`, one stalled process, abridged)

Thread A (a test thread):

    ntdll!RtlpAcquireSRWLockExclusiveContended
    CRYPT32!CDefaultChainEngineMgr::GetDefaultCurrentUserEngine
    CRYPT32!CertGetCertificateChain
    wintrust!WinVerifyTrust
    tbb12!...                       (inside DllMain)
    ntdll!LdrpCallInitRoutine
    ntdll!LdrLoadDll / KERNELBASE!LoadLibraryExA
    intelocl64!...clCreateContext
    ur_adapter_opencl!urContextCreate <- sycl9 context::context <- sycl9 queue::queue

Thread B (unnamed, started by the NVIDIA stack):

    ntdll!LdrpDrainWorkQueue          (waiting for the loader lock)
    ntdll!LdrpLoadForwardedDll <- LdrResolveDelayLoadedAPI <- KERNELBASE!_delayLoadHelper2
    KERNELBASE!RegisterGPNotificationInternal
    CRYPT32!CCertChainEngine::CCertChainEngine <- CDefaultChainEngineMgr::GetDefaultCurrentUserEngine (lock held)
    CRYPT32!CertGetCertificateChain <- wintrust!WinVerifyTrust
    nvdiagclt64!diagcltGetCallbacks+...

Full dumps are kept by kept locally (`C:/cb/bonefish40-level3-stacks.txt`, `...-mutex-stacks.txt`).

## Ask

- Intel: do not call `WinVerifyTrust` (or anything that can take CRYPT32 locks or load DLLs) from `DllMain` in `tbb12`.
- NVIDIA: `nvdiagclt64` should not build a CRYPT32 chain engine on a thread that can then block on the loader lock, or should not start it during OpenCL ICD load.
- Either fix alone removes the deadlock.

## Not shown

Which vendor is "at fault" is a judgement; both calls are legal alone. This was seen on one machine with both
Intel and NVIDIA OpenCL installed; whether it needs the NVIDIA stack was not tested on a machine without it.
