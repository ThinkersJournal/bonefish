# Bonefish for Fuel's Intel Arc A770 row (R5): interface proposal

Status: **design, no code**. Bonefish lane, 2026-10-31, on the PM's task (R5). The question it serves is Fuel ROADMAP row 127: *for the GEMM-class ops a model spends its time in, how far is Fuel's Vulkan backend from `oneapi::mkl::blas` on the same Arc A770, so the SYCL shim has a measured reason to exist or not.* Nothing here has run on a GPU. **Every GPU statement is UNVERIFIED**; the rig dual-boots, so Windows and Linux results are separate columns and each is UNVERIFIED until measured.

What Fuel said (Fuel 1, over the peer channel, 2026-10-31, "read-not-measured"; recorded as answers, not as measurements): crates.io deps only, so no path/git dependency on Bonefish; a Bonefish-built CLI printing JSON is the easiest thing to run on the rig; device time via event profiling plus a no-op launch floor (wall minus floor only as a labelled fallback); host half types are `half::f16` / `half::bf16` (half 2.7.1); compare f32-accumulate (UNVERIFIED on Fuel's side: the Vulkan shaders were not checked); weights are loaded transposed so a linear is row-major `x[M,K] @ W[K,N] -> y[M,N]`, NoTrans/NoTrans, alpha 1, beta 0; no stable device id exists on Fuel's side yet (row 125 will add PCI bus id + device UUID), so Bonefish must send what the Level Zero device exposes; dequantize is OUTSIDE the MKL timed region and reported as a separate row.

## 1. What row 127 asks of Bonefish

`gemm` and `gemv` (M = 1) via oneMKL SYCL on the A770, f32 and f16 (bf16 only if both stacks expose it, stated either way): decode matvec M=1, K=N=5120; prefill GEMM M in {128, 1024}, K=N=5120; an FFN-shaped K=5120, N=17408-class shape. Device-resident operands, host-to-device copy excluded, output read back once outside the timed region. Median device time over N timed runs after W warm-ups (declared up front), with min/max, ms, GFLOP/s (2MNK/time) and GB/s for the matvec. Controls: same harness against a CPU f64 reference (tolerance sabotage-calibrated; a deliberately wrong shape must fail), an empty-kernel timing for the launch floor, and the RTX 4060 through cuBLAS if available as a known-faster stack. Not claimed: anything about models, SYCL on other GPUs, Windows SYCL support for Arc.

## 2. What Bonefish has today, and the gaps

Checked in the tree at `22adfcf` (0.0.15) and in the oneAPI 2026.1 headers on the laptop.

| Need | Today | Gap |
|---|---|---|
| `gemm` f32 | yes, blocking and scoped-async, column-major, `UsmKind::Device` exists | none for f32 |
| `gemm` f16 / bf16 | none. `Element` is sealed to `f32`, `f64`, `Complex<_>`, `i64` | oneMKL's USM `gemm` is declared for (A,B,C,scalar) = (half,half,half,half), (half,half,float,float), (bf16,bf16,bf16,float), (bf16,bf16,float,float) (`blas/usm_decls.hpp`). **What accumulation precision the all-half variant uses is not stated in the header.** Whether the A770 + driver runs any of them: UNVERIFIED |
| `gemv` f16 | none | oneMKL declares **no half `gemv`** (grep of `usm_decls.hpp`): an M=1 f16 case is `gemm` with one row, labelled, which Fuel accepts |
| Row-major | column-major only | mapping below, or `oneapi::mkl::blas::row_major` |
| Pick a specific GPU | `DeviceKind::Gpu` = SYCL `gpu_selector_v` (the runtime's first GPU) | no enumeration, no way to say "the A770". Headers expose `sycl::ext::intel::info::device::{pci_address, uuid, device_id}` (gated on aspects); whether the A770's driver fills them: UNVERIFIED |
| Device time | none | needs `property::queue::enable_profiling` (exists in the headers) and `command_start` / `command_end` from the event |
| Launch floor | none | an empty-kernel submission with the same profiling |
| Delivery | `bonefish` is not on crates.io | Fuel takes crates.io deps only |

**Row-major mapping (derived, to be tested, not yet run).** A row-major `X[M,K]` is the column-major `K x M` matrix X^T; row-major `W[K,N]` is column-major `N x K` (W^T); row-major `Y[M,N]` is column-major `N x M` (Y^T). So `Y = X W` is `Y^T = W^T X^T`: one column-major `gemm` with `m = N, n = M, k = K`, A = the W buffer (`lda = N`), B = the X buffer (`ldb = K`), C = the Y buffer (`ldc = N`), NoTrans/NoTrans. A safe `blas::gemm_row_major` would do that swap and be tested against a naive reference, including a non-square shape where a wrong swap gives a different answer.

## 3. The contract Bonefish offers Fuel

These are stated so the harness does not rely on anything not measured.

1. **Ordering.** An in-order SYCL queue does **not** order oneMKL calls on the shipped build: `dot` after `axpy`/`copy`/`gemm` read stale data in 98 to 100 of 100 rounds (ASYNC-DESIGN section 8; measured on the CPU device, Windows, oneAPI 2026.1 only: the Linux numbers with the chain off were never observed). Bonefish therefore chains every asynchronous submission to the previous one. For a timing harness this means **timed runs are serial by construction**; a harness that wants overlap must say so and is on its own. Whether the same is true for `gemm` on a GPU is UNVERIFIED. (`gemm` after `gemm` was ordered on the CPU device; that says nothing about a GPU.)
2. **Errors.** Asynchronous errors are collected per queue and returned by the next wait that collects them; `Ok` from a wait is not proof the work succeeded, and an error can arrive from a later wait on the same queue (measured on the CPU device, about 2 to 5 in 100 with four threads active, 0 of 150 single-threaded). A blocking shim call behind a failing task waited for it and returned success in the one case measured (`bf_memcpy`, R3d); the other blocking calls share the same wait and were not measured individually. The blocking `blas::axpy`, `dot`, ... wrappers go through a scope and collect errors like it. The harness's correctness check (against the CPU f64 reference) is therefore not optional: it is what catches a wrong or failed result.
3. **Thread-safety.** Every oneMKL call takes one process-wide lock (the shipped oneMKL crashed under concurrent first use). A harness times one operation at a time, so the lock costs nothing there; it would serialise a multi-threaded caller.
4. **dtype coverage promised:** f32 and f64 exist today; f16 and bf16 `gemm` are proposed (section 4) with the accumulate variant named in every output row. No `gemv` for f16/bf16.
5. **Quantized.** oneMKL BLAS has no K-quant or Q4_0 kernel. Bonefish provides nothing quantized. The MKL row for a quantized weight is dequantize-then-GEMM with the dequantize **outside** the timed region (Fuel's choice) and is labelled "not a like-for-like kernel race".
6. **No peak figures.** The A770's peak numbers are not measured by Bonefish; if quoted, only with their source.

## 4. Proposal: four small steps, then the rig

| Step | What | Size | Acceptance (each has a negative control) |
|---|---|---|---|
| R5a | `Queue::devices()` listing name, backend, device type, and `pci_address` / `uuid` / `device_id` where the aspect exists (else `None`), plus `Queue::with_device(&info)`. | ~150 lines | on CPU: the CPU device is listed and selectable; a bogus index is an error; a device missing the aspect reports `None`, not a guess. GPU: UNVERIFIED |
| R5b | Profiling: opt-in `profiling` queue property, `Event::device_time()` (start/end in ns, after the wait), and an empty-kernel submission for the launch floor. | ~150 lines | on CPU: a `gemm` event on a profiling queue has `end >= start > 0`; the same call on a non-profiling queue is an error, not a zero; the empty kernel is faster than a 1024^2 `gemm` |
| R5c | f16 / bf16 `gemm` (`half::f16`, `half::bf16`, the crates.io release at the time) with the accumulate variant as an explicit argument, plus `blas::gemm_row_major`. | ~300 lines | row-major vs naive reference on non-square shapes (a swapped mapping must fail it); f16 vs f64 reference within a stated tolerance; **if the CPU device does not support half `gemm`, the test says so and is reported as not run, never skipped silently** |
| R5d | `bonefish-bench` CLI (workspace member, built only with oneAPI): declared shapes, N/W up front, prints JSON rows (shape, dtype, accumulate variant, median/min/max device ms, GFLOP/s, GB/s for the matvec, device id, library versions, OS) and verifies sampled output entries against an f64 reference. | ~300 lines | an injected wrong shape fails verification; the launch floor is printed next to each row; no GPU needed to build or to run on the CPU device |
| rig | run R5d on the A770 under each OS, paste into Fuel's dated note. | n/a | waits for the rig (CireSnave); every number carries the versions Fuel's ledger requires |

Sampled verification (a few hundred random output entries, each a full f64 dot product) instead of a full reference `gemm` keeps the check cheap at M=1024, K=N=5120 and is **less** than a full comparison; the CLI prints how many entries it checked.

## 5. Options for the choices that are not forced

**Delivery of the harness** (Fuel's Q1): (a) Fuel depends on `bonefish` from crates.io, which needs it published first (a PM / CireSnave decision, and the crate is pre-1.0 with a sealed, still-changing API); (b)/(c) a CLI in the Bonefish repo that Fuel runs and pastes. **Recommend (c)**: it matches what Fuel asked for, needs no publish, and keeps Fuel's rule of crates.io-only dependencies.

**Timing source:** (a) event profiling (device time; what row 127 specifies); (b) wall time of a blocking call minus the launch floor (includes host overhead and the submission path). **Recommend (a)**, with (b) printed only when profiling is unavailable on the device and labelled as the fallback. Whether the A770's driver supports profiling queues under each OS: UNVERIFIED.

**f16 type:** (a) depend on `half` (the type Fuel uses; `repr(transparent)` over `u16`, the same layout as `sycl::half` and the bf16 type); (b) own newtypes over `u16`. **Recommend (a)** at the newest crates.io release when implemented (CireSnave's dependency rule); the layout equality is the assumption to test, not to trust.

**Accumulate precision:** report the oneMKL variant used in every row: `(half,half,half,half)` (internal accumulation unstated), `(half,half,float,float)` (f32 scalars and f32 output, so not an f16-output race), and the bf16 pair. Fuel compares f32-accumulate (UNVERIFIED on Fuel's side); the row that matches is the one to compare, and the others are reported but not ranked against Fuel's.

## 6. Not measured, not claimed

- Anything on a GPU: whether the A770 + driver exposes half or bf16 `gemm`, profiling, PCI address, UUID; performance; correctness. Windows SYCL support for Arc is not measured.
- Whether `gemm` is ordered by an in-order queue on a GPU (only the chain makes serial timing safe; the cost of the chain on a GPU is unmeasured).
- The accumulation precision of oneMKL's all-half `gemm`.
- That `half::f16` and `sycl::half` have identical bit layouts on every path (expected from the types, to be tested).
- Anything about Fuel's Vulkan shaders (Fuel has not checked them either).

## 7. Asks back to Fuel

1. Confirm the CLI output fields (section 4, R5d) cover your ledger columns, or list what is missing.
2. Confirm the row-major mapping in section 2 matches your matmul's (A, B) order, or give me the exact call to mirror.
3. When row 125's selector lands, tell me the id format so `Queue::with_device` can take it directly.
