# bonefish

Planned. The crate names `bonefish` and `bonefish-sys` are reserved on crates.io. No code yet; see `ROADMAP.md`.

Licensed under MIT or Apache-2.0, at your option.

## Patent notes

The SYCL emitter targets non-NVIDIA hardware and does not lower NVIDIA-patented operations to hardware operations; it refuses them by name (typed `UnsupportedKernel` error). Declined: async copy / commit / wait and collective barrier-reductions (US12118382B2, US11294713B2, US8539204B2), and the mbarrier statements `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `Tcgen05Commit`, `CopyBulk` (US11803380B2). Not legal advice; detail and limits in `ROADMAP.md` row R13.
