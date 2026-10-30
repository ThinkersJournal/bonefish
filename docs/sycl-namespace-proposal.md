# Proposal: the `sycl:` capability-set vocabulary

Status: **proposal from the Bonefish lane to KISS**, 2026-10-30. Not registered; KISS records the vocabulary (KISS-CLASSIFY-6.8-0004) and the PM routes it. Model: `spec/namespaces/cuda.md`. Maintainer would be Bonefish; reference implementation `bonefish-sycl-emit::target`.

Measured against DPC++ 2026.1 (`icpx` 2026.1.0.20260617, Windows) and its shipped table `sycl/ext/oneapi/experimental/device_architecture.def`. Compile checks only: no Intel GPU was available, so nothing here was run on one.

## 1. What a token names

As for `cuda:`, a token names **the specialization a kernel was built for**, not the maximum capability of the part it runs on. Matching is byte-exact (§6.8-0002); there is no range or implication.

## 2. Grammar

```text
sycl:spir64                   portable: JIT-compiled SPIR-V, the default `icpx -fsycl` target
sycl:intel-gpu-<arch>         ahead-of-time for one Intel GPU architecture
```

- `<arch>` is the **canonical** name of an `intel_gpu_*` entry in the table with every `_` replaced by `-`. The token is the `-fsycl-targets` value spelled with hyphens.
- After the single `:` the field is lowercase `[a-z0-9-]`. No `.` fields, no member list, no range: a **single scalar**, as `cuda:` is, so §6.8-0006 (fixed-width juxtaposition) is vacuous and §6.8-0007 (digest fallback) is unreachable.
- **Aliases get no token.** `intel_gpu_dg2_g10` is another name for `intel_gpu_acm_g10` (same value in the table); a second token would split one cell in two under byte-exact matching. The table marks them with `__SYCL_ARCHITECTURE_ALIAS`; 48 of the 143 entries are aliases.

## 3. Token set (36 tokens)

`sycl:spir64`, and one `sycl:intel-gpu-<arch>` for each canonical Intel GPU entry: the 34 of the 2026.1 table, `bdw skl kbl cfl apl glk whl aml cml icllp ehl tgllp rkl adl-s adl-p adl-n dg1 acm-g10 acm-g11 acm-g12 pvc pvc-vg mtl-u mtl-h arl-h bmg-g21 bmg-g31 lnl-m ptl-h ptl-u wcl nvl-s nvl-u nvl-p`, plus `cri`, which the 2026.2 table adds (seen on the Linux CI runner, which installs 2026.2 from Intel's apt repository; absent from the 2026.1 table on the Windows laptop).

**The set grows with the compiler.** A new architecture is an additive spelling (as for `cuda:`), so each token records the first DPC++ release that produces it (`TOKENS` pairs). The gate checks a token's witness only against a compiler at or after that release, and checks that the installed table yields nothing the list lacks. A list pinned to one release cannot pass on both: this was found when the first version of the gate failed on 2026.2 for `intel_gpu_cri`. The authoritative list is `bonefish-sycl-emit/src/target.rs` (`TOKENS`), which the gate checks against the installed table in both directions.

Deliberately **out of scope**: NVIDIA and AMD entries of the same table (`nvidia_gpu_*`, `amd_gpu_*`) belong to `cuda:` and the HIP lane's namespace, and Intel CPU entries (`intel_cpu_*`, `x86_64`) are not built for in v1.

## 4. Derivability witness (§6.8-0014)

| Entry | Witness (upstream construct) |
|---|---|
| `sycl:intel-gpu-<arch>` | the `__SYCL_ARCHITECTURE(intel_gpu_<arch>, ...)` line (not `_ALIAS`) in `device_architecture.def`, which also declares `sycl::ext::oneapi::experimental::architecture::intel_gpu_<arch>` |
| `sycl:spir64` | `-fsycl-targets=spir64`, accepted by `icpx` (the default target) |

Direction: the witness says every **named** capability is **reachable** in the table; it does not say the Bonefish emitter's output has been run on that architecture. Only `intel_gpu_acm_g10` was additionally compiled ahead-of-time here (compile-only); the rest are table-derived and unexercised.

**Upstream, not lateral or downstream.** The table is DPC++'s own, which our emitter does not generate; a token list restated from our code would be downstream and invalid.

## 5. The gate

`bonefish-sycl-emit/tests/sycl_target_tokens.rs`, run in the `oneapi` CI job with `BONEFISH_REQUIRE_SYCL=1` so a missing toolchain fails rather than skips:

- `every_named_token_is_produced_by_the_installed_table`: each token has an upstream entry (the 6.8-0014 direction).
- `the_vocabulary_is_complete_for_the_installed_table`: the table yields no token the list lacks (drift).
- `the_gate_fails_for_the_reason_the_clause_names`: deleting the `intel_gpu_pvc` line makes the witness check report exactly `sycl:intel-gpu-pvc`, and the unmutated table reports nothing. This is the failure the clause names, not an adjacent one (such as a parse error).
- `the_compiler_accepts_the_named_targets`: `spir64` and `intel_gpu_acm_g10` compile; `intel_gpu_acm_g10_not_a_target` is rejected (negative control).

No compile-time binding exists: the table is a C++ X-macro header, so the witness is evaluated at test time against the installed compiler.

## 6. Questions for KISS

1. Is `-` acceptable in a capability-set field? `vulkan:` uses it; `cuda:` does not.
2. `cuda:` admits a *set* per device (`sm89` and the `sm80` fallback). For SYCL I would claim a device admits its own `sycl:intel-gpu-*` token **and** `sycl:spir64` (JIT runs anywhere SYCL does). The second half is by design of SPIR-V JIT and **unmeasured on a GPU**; say if KISS wants that kept out until it is.
3. `unpopped-vocab`'s `ArchSku` is a Rust enum (`Sm80`, `Sm89`, ...) with no SYCL variant. Whether this vocabulary becomes variants there, or the target stays a free token, is for the Unpopped lane.
