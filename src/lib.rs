//! Bonefish: a safe Rust interface to Intel oneAPI / oneMKL on SYCL devices.
//!
//! - [`Queue`]: a SYCL queue on one device ([`DeviceKind`]).
//! - [`UsmBuffer`]: typed unified shared memory owned by a queue.
//! - [`blas::gemm`]: matrix multiply, `f32` and `f64`; level 1 and `gemv` ([`blas::axpy`],
//!   [`blas::dot`], [`blas::nrm2`], ...), blocking.
//! - [`Dft`]: complex-to-complex FFTs, `f32` and `f64`, 1-D and N-D.
//!
//! The plain operations are blocking. For work that runs while the host continues, see
//! [`Queue::scope`]: inside a scope, operations return an [`Event`] immediately and the scope
//! waits for everything on its way out, so the borrow checker can keep the host away from buffers
//! the device is still using. Every buffer extent is checked in Rust before the device is
//! touched, so a safe call cannot read or write outside a [`UsmBuffer`].
//!
//! # Building without oneAPI
//!
//! The crate always compiles. When oneAPI's `icpx` was not found at build time the same API is
//! present, [`SYCL_AVAILABLE`] is `false`, and [`Queue::new`] returns [`Error::Unavailable`], so
//! a downstream crate can depend on this one unconditionally and decide at run time.
//!
//! # What has been run
//!
//! Only on the CPU (OpenCL) device, on Windows and Linux. No Intel GPU has run any of this.

pub mod blas;
mod device;
mod dft;
mod error;
mod queue;
mod scalar;
mod scope;
mod usm;

pub use blas::Transpose;
pub use device::{DeviceInfo, DeviceType, find_by_pci_address, find_by_uuid};
pub use dft::{Dft, Placement};
pub use error::{Error, Result};
pub use queue::{DeviceKind, Queue};
pub use scalar::{Complex, Element, Scalar};
pub use scope::{Event, Operand, Plan, Read, Scope, Write};
pub use usm::{UsmBuffer, UsmKind};

/// Whether the SYCL shim was built into this binary.
pub const SYCL_AVAILABLE: bool = bonefish_sys::SYCL_AVAILABLE;
