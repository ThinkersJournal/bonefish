//! Stand-ins with the same signatures, by operation family (R15). Only compiled without oneAPI.

mod base;
mod dft;
mod level1;
mod level2;
mod level3;

pub use base::*;
pub use dft::*;
pub use level1::*;
pub use level2::*;
pub use level3::*;
