# Bonefish roadmap

Order matters: survey, then design, then code. Every code row ships with a negative control (a test that fails when the behaviour is removed). Statuses: todo, doing, review (PR open), done.

| Row | What | Status |
|-----|------|--------|
| R0 | `ROADMAP.md`, CI (fmt, clippy, test, doc, cargo-deny, portfolio), then branch protection requiring them (PM) | done |
| R1 | Survey and design note `docs/SURVEY.md`: options first, recommendation, no code. Covers the oneMKL SYCL interface (`oneapi::mkl::blas`, `dft`), the C++ shim shape, Level Zero, Linux and Windows builds, and what a parser and emitter for SYCL/DPC++ kernels owes Unpopped's contract | done |
| R2 | `bonefish-sys`: `extern "C"` shim over oneMKL SYCL built with `icpx -fsycl`, no `onemkl-sys` dependency (PM ruling 2026-10-30). Done: queue, USM, blas gemm (s, d), complex DFT. Measured on CPU/OpenCL devices only (Windows laptop, Linux CI); GPU UNVERIFIED. Left: rest of blas, real-domain DFT, GPU runs once hardware is available | review |
| R3 | `bonefish`: safe queue, buffer and BLAS API, then `dft` | todo |
| R4 | SYCL/DPC++ parser and emitter on Unpopped's contract (`parser-emitter-ownership-map.md`, Unpopped ROADMAP 6.7) | todo |
| R5 | Intel GPU backend design with the Fuel lane (Fuel ROADMAP row 127: A770, Vulkan vs oneMKL via SYCL) | todo |
| R6 | Take over the `onemkl` / `onemkl-sys` ROADMAP, starting with PR #1 in `ciresnave/onemkl` (pending CireSnave's decision) | blocked |
