# Bonefish lane HANDOFF (new lane, created 2026-10-30 by the PM)

You own **Bonefish**: a comprehensive Rust interface to Intel oneAPI / oneMKL on Intel GPUs (DPC++/SYCL), plus the existing oneMKL crates. Crates reserved on crates.io at 0.0.0: `bonefish`, `bonefish-sys` (owner ciresnave). Repos: ThinkersJournal/bonefish (this checkout, `C:\Projects\bonefish`) and the existing `ciresnave/onemkl` (checkout `C:\Projects\onemkl`, crates `onemkl` + `onemkl-sys`, the CPU interface; version 0.2.0; PR #1 there withdraws the "SYCL out of scope" line, in the gate). PM proposal awaiting CireSnave (board 329): you own oneMKL's CPU interface too, one vendor SDK.

CireSnave, 2026-10-30, verbatim: *"Intel has some extremely promising GPU hardware coming out soon that is supposed to support 400+ GB of RAM.  I'd like to get ahead on supporting Intel hardware before that hits."* *"I doubt any existing ML design has pushed OneMKL to its limits as it is often supported as an afterthought since the industry has focused on CUDA.  I'd like to have comprehensive ROCm, and oneMKL support built for Fuel and for Unpopped so we can really test the limits of every manufacturer's software stack."* *"I don't have a better option than bONEfish.  Go with that."* *"Bonefish should own SYCL/DPC++ and create the parser and emitter for that."*

## Facts the PM measured
oneAPI 2026.1 is installed on the laptop at `C:\Program Files (x86)\Intel\oneAPI\`: oneMKL C headers (`mkl/latest/include/mkl*.h`), the SYCL interface (`include/oneapi/mkl/blas.hpp`, `dft.hpp`: `oneapi::mkl::*` taking a `sycl::queue`), and the DPC++ compiler `compiler/2026.1/bin/icpx.exe`. DPC++ is Intel's SYCL implementation (`icpx -fsycl`); SYCL is a Khronos standard. The SYCL path needs a small C++ shim exposing `extern "C"` over `oneapi::mkl::blas` (then `dft`), built with icpx, depending on `onemkl-sys` from crates.io. Hardware: the desktop rig has an Intel Arc A770; CireSnave expects very large-memory Intel GPUs soon. Fuel ROADMAP row 127 (on fuel main) is the A770 comparison, Vulkan vs oneMKL-via-SYCL; oneMKL BLAS has no K-quant kernels so quantized comparisons are dequantize+GEMM, labelled not like-for-like. **Verify each claim yourself.**

## Scope
- Intel GPU computing through oneAPI (SYCL, Level Zero as needed), comprehensive, Linux AND Windows.
- Parser + emitter for SYCL/DPC++ kernels on Unpopped's contract (read `C:\Projects\Unpopped\docs\parser-emitter-ownership-map.md`, ROADMAP 6.7). Unpopped holds neither.
- Eventually the Fuel backend for Intel; design with the Fuel lane; do not edit its repo.
- The onemkl crates: take over their ROADMAP; first, the open PR #1.

(see COMMON rules below)
## Rules that bite here
Read `C:\Projects\CLAUDE.md` (loaded) and `C:\Projects\AGENT-ONBOARDING.md` once. Work in a worktree (`C:\Projects\<repo>-wt-<topic>`), never branch in the primary checkout. Version bump on every pushed change (pre-1.0 breaking = the SECOND number); the PM allocates the number at gate time. crates.io versions only, no path or git dependencies; all dependencies on their latest versions. Trace and verify every third-party file against its origin and keep its licence and credit (CireSnave: "I am not a plageurist"). Every claim verified on the forge or by measurement; every absence needs a positive control. Compact `[READY]`/`[STATUS]` messages, short. Peers change ID on every restart: `list_peers`, match on cwd. Restart at task boundaries (HANDOFF rewrite, `lane-restart assert-idle` last, then `--self`). This is an open-source project from day one: no secrets, no private paths in committed files.

## First deliverables (all lanes of this family)
0. `ROADMAP.md` with rows and statuses; a CI workflow (fmt, clippy, tests, cargo-deny, and the portfolio checks workflow the other repos use: copy the shape from `C:\Projects\capant\.github`) in your first PR; then `[ASK]` the PM to enable branch protection requiring those checks (CI and protection everywhere).
1. A survey and design note (options first, recommendation, no code) before any crate code.
