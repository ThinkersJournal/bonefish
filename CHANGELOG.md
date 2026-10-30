# Changelog

## 0.0.2

- `bonefish-sys`: C shim over oneMKL SYCL (queues, USM, column-major sgemm/dgemm), built with `icpx -fsycl` when oneAPI is found; non-required `oneapi-linux` CI job. Measured on CPU/OpenCL devices only (Windows laptop, Linux CI); GPU UNVERIFIED. Both crates stay `publish = false`.

## 0.0.1

- Docs: R1 survey and design note (`docs/SURVEY.md`); ROADMAP R0 done, R1 in review. No code change; stub crate stays `publish = false`.
