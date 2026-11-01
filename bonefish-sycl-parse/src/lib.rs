//! The SYCL/DPC++ front end for [Unpopped](https://crates.io/crates/unpopped).
//!
//! Bonefish owns SYCL in the CireSnave ecosystem (the emitter is `bonefish-sycl-emit`), so the
//! language's parser lives here. Unpopped keeps the language-neutral half: [`Frontend`] and the
//! `lift_*` walkers in `unpopped::convert`, which this crate feeds with a grammar binding, the way
//! `baracuda-cuda-parse` does for CUDA and `vulkane-slang-parse` for Slang.
//!
//! * [`parse_sycl`] - SYCL (C++) source to an error-tolerant tree-sitter CST (`tree-sitter-cpp`).
//! * [`SYCL_RESIDUE`] - constructs refused as inexpressible (local memory, barriers, atomics,
//!   sub-groups, library calls).
//! * [`SYCL`] - the [`Frontend`]: a `parallel_for` kernel body storing `out[i] = <expr>` over
//!   `in{K}[i]` operands, with `sycl::` math calls.
//!
//! # The idiom set (v1)
//!
//! The shared walkers recognise `out[i] = <expr>;` where `<expr>` is arithmetic over `in{K}[i]`,
//! literals, ternaries, `sycl::` math calls and local bindings. That is exactly the shape
//! `bonefish-sycl-emit` generates, and it is what a hand-written SYCL `parallel_for` over USM
//! pointers named `out`/`in0`/`in1` looks like. The buffers may instead be declared with a
//! `ResourceMap` (`lift_elementwise_with`), which renames them. Reductions and scans use the same
//! naive accumulator-loop idioms as the other front ends.
//!
//! # Math spellings: `sycl::` only, on purpose
//!
//! `sycl::exp(x)` lifts to `Exp`; the libm names (`expf`, `sinf`, ...) are **refused as
//! unrecognised**. Measured (ROADMAP R7, `docs/r7/RESULTS.md`; icpx 2026.1 and 2026.2, CPU
//! devices): under the compiler's default floating-point model `expf` is off by up to 7620 ULP,
//! `sinf`/`cosf` by about 5000, `tanf` by 1579, while `sycl::` spellings stay within 4.3 ULP. A
//! front end that read `expf` as the accurate `Exp` would assert an accuracy the source does not
//! have, and the neutral IR cannot carry compiler flags; so the source must say `sycl::exp`.
//!
//! ```
//! use bonefish_sycl_parse::SYCL;
//! use unpopped::convert::lift_elementwise;
//! use unpopped_vocab::ElementKind;
//!
//! let src = "q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { size_t i = id[0]; out[i] = in0[i] * in1[i]; });";
//! let lifted = lift_elementwise(&SYCL, src, "mul", &[ElementKind::F32]).unwrap();
//! assert_eq!(lifted.n_inputs, 2);
//! ```
//!
//! The grammar crate (`tree-sitter-cpp`) is named in exactly one place, [`parse_sycl`].

use tree_sitter::{Parser, Tree};
use unpopped::convert::Frontend;
use unpopped::ir::{BinaryOp, UnaryOp};

/// Parse SYCL source into a tree-sitter CST.
///
/// SYCL is C++, so this is the C++ grammar; `tree-sitter-cpp` has the node and field names the
/// shared walkers read (`subscript_expression`'s `indices`). `None` only if the grammar fails to
/// load or the parser is cancelled; syntax errors do not return `None`, they appear as `ERROR`
/// nodes in the tree.
#[must_use]
pub fn parse_sycl(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
        .ok()?;
    parser.parse(src, None)
}

/// SYCL constructs that aren't IR-expressible: work-group local memory, barriers and fences,
/// atomics, sub-group operations, SYCL reductions, and oneMKL library calls.
pub const SYCL_RESIDUE: &[&str] = &[
    "local_accessor",
    "group_barrier",
    "atomic_fence",
    "atomic_ref",
    "sub_group",
    "joint_",
    "sycl::reduction",
    "oneapi::mkl",
];

/// A one-argument `sycl::` math call to the IR op it computes. Only the qualified spelling is
/// recognised (see the crate docs): `expf` and a bare `exp` are `None`.
fn sycl_unary(name: &str) -> Option<UnaryOp> {
    Some(match name.strip_prefix("sycl::")? {
        "exp" => UnaryOp::Exp,
        "exp2" => UnaryOp::Exp2,
        "log" => UnaryOp::Log,
        "sqrt" => UnaryOp::Sqrt,
        "rsqrt" => UnaryOp::Rsqrt,
        "fabs" | "abs" => UnaryOp::Abs,
        "tanh" => UnaryOp::Tanh,
        "sin" => UnaryOp::Sin,
        "cos" => UnaryOp::Cos,
        "erf" => UnaryOp::Erf,
        "floor" => UnaryOp::Floor,
        "ceil" => UnaryOp::Ceil,
        "trunc" => UnaryOp::Trunc,
        _ => return None,
    })
}

/// A two-argument `sycl::` math call to the IR op. `sycl::fmax` / `sycl::fmin` follow C `fmax`
/// (a NaN operand is ignored), the IR's `FmaxIeee` / `FminIeee`; `sycl::max` / `sycl::min` are
/// deliberately absent, because the SYCL spec leaves their NaN behaviour unspecified.
fn sycl_binary(name: &str) -> Option<BinaryOp> {
    Some(match name.strip_prefix("sycl::")? {
        "fmax" => BinaryOp::FmaxIeee,
        "fmin" => BinaryOp::FminIeee,
        "pow" => BinaryOp::Pow,
        "atan2" => BinaryOp::Atan2,
        "copysign" => BinaryOp::Copysign,
        _ => return None,
    })
}

/// SYCL: `parallel_for` kernels, `out` / `in{K}` buffers, `sycl::` math.
pub const SYCL: Frontend = Frontend::new(
    "sycl",
    "parallel_for",
    SYCL_RESIDUE,
    "out",
    "in",
    parse_sycl,
)
.with_intrinsics(sycl_unary, sycl_binary);
