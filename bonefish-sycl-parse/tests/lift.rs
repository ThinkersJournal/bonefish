//! SYCL front end: hand-written kernels, the refusals, and a round trip through the emitter.
//! Toolchain-free: nothing here compiles SYCL.

use bonefish_sycl_emit::Sycl;
use bonefish_sycl_parse::{SYCL, SYCL_RESIDUE, parse_sycl};
use unpopped::LiftError;
use unpopped::convert::{lift_elementwise, lift_reduction};
use unpopped::generate;
use unpopped::ir::{OpDef, ScalarExpr, input};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: &[ElementKind] = &[ElementKind::F32];

fn kernel(body: &str) -> String {
    format!(
        "q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {{\n  size_t i = id[0];\n  {body}\n}});"
    )
}

#[test]
fn the_grammar_loads_and_broken_source_is_an_error_node_not_none() {
    let tree = parse_sycl("int main() { return 0; }").expect("grammar loads");
    assert!(!tree.root_node().has_error());
    let bad = parse_sycl("void f( {").expect("a tree even for broken source");
    assert!(bad.root_node().has_error());
}

#[test]
fn a_multiply_lifts() {
    let l = lift_elementwise(&SYCL, &kernel("out[i] = in0[i] * in1[i];"), "mul", F32).unwrap();
    assert_eq!(l.n_inputs, 2);
    assert_eq!(l.op.body, (input(0) * input(1)).0);
}

#[test]
fn sycl_math_lifts_and_libm_is_refused_as_unrecognised() {
    let ok = lift_elementwise(&SYCL, &kernel("out[i] = sycl::exp(in0[i]);"), "e", F32).unwrap();
    assert!(
        matches!(ok.op.body, ScalarExpr::Unary(..)),
        "{:?}",
        ok.op.body
    );
    // Same expression, libm spelling: refused, because `expf` under default flags is not `Exp`.
    for libm in ["expf(in0[i])", "sinf(in0[i])", "exp(in0[i])"] {
        let r = lift_elementwise(&SYCL, &kernel(&format!("out[i] = {libm};")), "e", F32);
        assert!(
            matches!(r, Err(LiftError::Unrecognized(_))),
            "{libm}: {r:?}"
        );
    }
}

#[test]
fn every_residue_marker_is_refused_as_inexpressible() {
    for marker in SYCL_RESIDUE {
        let src = kernel(&format!("{marker}; out[i] = in0[i];"));
        let r = lift_elementwise(&SYCL, &src, "x", F32);
        assert!(
            matches!(r, Err(LiftError::Inexpressible(ref w)) if w == marker),
            "{marker}: {r:?}"
        );
    }
}

#[test]
fn source_without_parallel_for_is_not_a_kernel() {
    let r = lift_elementwise(&SYCL, "void f(float* out) { out[0] = 1.0f; }", "x", F32);
    assert!(matches!(r, Err(LiftError::NotAKernel)), "{r:?}");
}

#[test]
fn a_naive_sum_loop_lifts_as_a_reduction_and_not_as_elementwise() {
    let src = "q.single_task([=]() { float acc = 0.0f; for (size_t i = 0; i < n; i++) acc += in0[i]; out[0] = acc; }); // parallel_for";
    assert!(lift_reduction(&SYCL, src, "sum", F32).is_ok());
    assert!(lift_elementwise(&SYCL, src, "sum", F32).is_err());
}

/// What the emitter generates parses back to the op it was generated from.
#[test]
fn the_emitters_output_round_trips() {
    let n = 7_i64;
    let d = OperandDesc::new(1, &[n], &[1], ElementKind::F32, 256);
    let ops = [
        (OpDef::elementwise("rt_add", 2, F32, input(0) + input(1)), 2),
        (OpDef::elementwise("rt_mul", 2, F32, input(0) * input(1)), 2),
        (OpDef::elementwise("rt_sub", 2, F32, input(0) - input(1)), 2),
        (OpDef::elementwise("rt_div", 2, F32, input(0) / input(1)), 2),
        (OpDef::elementwise("rt_exp", 1, F32, input(0).exp()), 1),
    ];
    for (op, n_in) in ops {
        let operands = vec![d; n_in + 1];
        let cat = if n_in == 1 {
            OpCategory::UnaryElementwise
        } else {
            OpCategory::BinaryElementwise
        };
        let key = structure_key(cat, &operands, ArchSku::Sm89);
        let k = generate(&op, &key, &Sycl);
        let lifted = lift_elementwise(&SYCL, &k.source, &op.name, F32)
            .unwrap_or_else(|e| panic!("{}: {e:?}\n{}", op.name, k.source));
        assert_eq!(lifted.op.body, op.body, "{}\n{}", op.name, k.source);
    }
}
