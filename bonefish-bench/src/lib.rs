//! Benchmark harness for Bonefish `gemm` (R5d, `docs/FUEL-INTEL-GPU-DESIGN.md`).
//!
//! Times row-major `y[M,N] = x[M,K] @ W[K,N]` (alpha 1, beta 0, no transposes; the layout Fuel
//! asked for) through the safe API: f32, f64 and the four half/bf16 variants, optionally as a
//! strided batch. Prints **one JSON object per line**. Every row carries the predicate of its
//! numbers: shape, dtype and accumulate variant, the number of warm-ups `W` and timed runs `N`
//! (declared up front), median/min/max, the launch floor, the device and its driver version, the
//! OS, the exact command line, and the verification result as the maximum error against an f64
//! reference with the tolerance used (not just pass/fail).
//!
//! Timing source is device time from event profiling. If the device cannot report it, the row says
//! `wall_minus_floor` instead and the number includes host overhead. **No speed is asserted
//! anywhere**: the tests check that rows are well formed and that verification fails on a wrong
//! layout, never that anything is fast.
//!
//! Verification is sampled: `--samples` random output entries, each a full f64 dot product over
//! the operands as the device saw them (already rounded to the input type). That is **less** than
//! comparing every entry, and the row says how many it checked.

mod cli;
mod run;

pub use cli::{Parsed, cli, parse_args, row_json};
pub use run::{Plan, Res, Row, Shape, Target, Timing, Variant, Verification, pick_device, run_row};
