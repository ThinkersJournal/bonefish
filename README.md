# bonefish

Rust interface to Intel oneAPI / oneMKL on SYCL devices (`bonefish`, over the `bonefish-sys` C++ shim), plus a SYCL/DPC++ parser (`bonefish-sycl-parse`) and emitter (`bonefish-sycl-emit`) on Unpopped's contract. Measured on CPU/OpenCL devices only; every GPU statement is UNVERIFIED. The crate names `bonefish` and `bonefish-sys` are reserved on crates.io. Status and open work: `ROADMAP.md`.

Licensed under MIT or Apache-2.0, at your option.

## Patent notes

The SYCL emitter targets non-NVIDIA hardware and does not lower NVIDIA-patented operations to hardware operations; it refuses them by name (typed `UnsupportedKernel` error). Declined: async copy / commit / wait and collective barrier-reductions (US12118382B2, US11294713B2, US8539204B2), and the mbarrier statements `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `Tcgen05Commit`, `CopyBulk` (US11803380B2), and `WgmmaWait`, a wait on a count of pending matrix-multiply operations (US12204897B2 and its continuation US12710961B2). Not legal advice; detail and limits in `ROADMAP.md` row R13.
