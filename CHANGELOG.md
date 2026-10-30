# Changelog

## 0.0.4

- New crate `bonefish-sycl-emit`: SYCL/DPC++ `Backend` for `unpopped` 0.34.11 (pinned exactly). v1 scope: contiguous elementwise, f32/f64, single output; everything else is a typed decline. Math is spelled as `sycl::` calls because icpx's default fast model compiles `expf` to a result 1.8e-4 off. Verified by compiling with icpx and running on the CPU/OpenCL device against `unpopped::oracle`; GPU UNVERIFIED. `publish = false`.

## 0.0.3

- `bonefish-sys`: complex-to-complex DFT (`bf_dft_c2c_new`, `bf_dft_compute`, `bf_dft_free`): single and double precision, 1-D and N-D, in-place and out-of-place, forward/backward scales. Shim split into `shim.cpp`, `dft.cpp`, `common.hpp`. CPU/OpenCL device only; GPU UNVERIFIED. Real-domain transforms not yet covered.

## 0.0.2

- `bonefish-sys`: C shim over oneMKL SYCL (queues, USM, column-major sgemm/dgemm), built with `icpx -fsycl` when oneAPI is found; non-required `oneapi-linux` CI job. Measured on CPU/OpenCL devices only (Windows laptop, Linux CI); GPU UNVERIFIED. Both crates stay `publish = false`.

## 0.0.1

- Docs: R1 survey and design note (`docs/SURVEY.md`); ROADMAP R0 done, R1 in review. No code change; stub crate stays `publish = false`.
