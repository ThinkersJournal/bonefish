//! SYCL/DPC++ backend for the neutral `unpopped` kernel generator.
//!
//! [`Sycl`] lowers a [`KernelPlan`] to SYCL C++ source: one `extern "C"` host function that
//! submits a `parallel_for` on a caller-provided `sycl::queue`. Modelled on the normative
//! reference emitter `unpopped-cpu-c` (0.34.11), and like it deliberately narrow in v1.
//!
//! # v1 scope (honest boundary)
//!
//! - Dtypes: `f32` and `f64` ([`Backend::supports_dtype`]). Everything else is a typed decline.
//! - Schedule: [`Schedule::Scalar`] only (the contiguous, single-output elementwise cell).
//!   Every other schedule is [`LowerError::UnsupportedSchedule`].
//! - Body nodes: input, constant, parameter, `+ - * /`, unary, binary, select, fma. A body with
//!   casts, bitcasts, qualifiers, locals, coordinates, reductions or packs is refused
//!   ([`LowerError::UnsupportedOp`]) rather than lowered through a default seam that panics.
//!
//! # The generated ABI
//!
//! ```text
//! extern "C" int NAME(void* queue, const T* in0, ..., T* out, long long n, T p0, ...);
//! ```
//!
//! `queue` is a `sycl::queue*`; every pointer must be USM reachable from that queue. The return
//! value is `0` on success, `1` for a SYCL exception, `2` for any other exception, `3` when a
//! `double` kernel is submitted to a device without `aspect::fp64`. `n` counts elements
//! ([`Backend::effective_count_width`] is the default `1`). This ABI is Bonefish's own: the
//! `GeneratedKernel` type carries only a name and source, so nothing in the contract describes
//! the extra queue argument or the status return yet.
//!
//! # Floating-point semantics
//!
//! Intel's compilers default to a fast floating-point model. The neutral IR requires NaN
//! propagation through `Max`/`Min` and spells it as a longhand ternary; the end-to-end tests
//! compile with the default flags *and* run a NaN case, so a compiler that folded that ternary
//! away would fail them rather than silently change the answer.

use unpopped::backend::{
    Backend, Decline, DeclinedOp, DtypeSupport, GeneratedKernel, LowerError, Lowering, Spelling,
    const_lit, lower_dag,
};
use unpopped::cfamily::{
    binary_f32, binary_f64, dtype_tag, out_ctype_of, param_args, param_ctype, scalar_ctype,
    select_f32, select_f64, store_expr_of, unary_f32, unary_f64,
};
use unpopped::ir::{BinaryOp, DagNode, ExprDag, ScalarExpr, UnaryOp};
use unpopped::plan::{KernelPlan, Schedule};
use unpopped_vocab::{ElementKind, TargetId};

/// The SYCL/DPC++ backend. Lowers a [`KernelPlan`] to SYCL C++ source.
#[derive(Copy, Clone, Debug, Default)]
pub struct Sycl;

impl Backend for Sycl {
    fn name(&self) -> &str {
        "sycl"
    }

    fn provider(&self) -> &str {
        "bonefish"
    }

    fn supports_dtype(&self, dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        // The answer is by dtype alone: `unpopped-vocab` 0.34.11 has no SYCL target namespace,
        // so there is nothing to condition on. `f64` is spelled correctly, but a device may lack
        // fp64; the generated host function reports that at run time (status 3) instead of
        // crashing. Exhaustive on purpose (`ElementKind` is not `#[non_exhaustive]`): a new
        // dtype is a build error here, and someone has to say whether this emitter computes it.
        match dtype {
            ElementKind::F32 | ElementKind::F64 => DtypeSupport::Supported,
            ElementKind::F32Strict
            | ElementKind::I8
            | ElementKind::U8
            | ElementKind::I16
            | ElementKind::U16
            | ElementKind::I32
            | ElementKind::U32
            | ElementKind::I64
            | ElementKind::U64
            | ElementKind::Bool
            | ElementKind::Fp8E4M3FN
            | ElementKind::Fp8E5M2
            | ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::I4
            | ElementKind::U4
            | ElementKind::B1
            | ElementKind::Complex64
            | ElementKind::Complex128
            | ElementKind::F8E8M0
            | ElementKind::F8E6M2
            | ElementKind::Fp8E4M3FNUZ
            | ElementKind::Fp8E5M2FNUZ
            | ElementKind::F4E2M1
            | ElementKind::F6E2M3
            | ElementKind::F6E3M2 => DtypeSupport::Unsupported,
        }
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        if !self
            .supports_dtype(plan.dtype, plan.key.target)
            .is_supported()
        {
            return Err(LowerError::UnsupportedDtype {
                dtype: plan.dtype,
                detail: "sycl backend v1: f32 and f64 only (see `Sycl::supports_dtype`)"
                    .to_string(),
            });
        }
        if plan.out_dtype != plan.dtype {
            return Err(LowerError::UnsupportedPlanShape {
                detail: format!(
                    "sycl backend v1: output dtype must equal the compute dtype; op '{}' has \
                     {:?} -> {:?}",
                    plan.op_name, plan.dtype, plan.out_dtype
                ),
            });
        }
        if plan.n_outputs != 1 {
            return Err(LowerError::UnsupportedPlanShape {
                detail: format!(
                    "sycl backend v1: single-output only; op '{}' has {} outputs",
                    plan.op_name, plan.n_outputs
                ),
            });
        }
        match plan.schedule {
            Schedule::Scalar => {}
            other => {
                return Err(LowerError::UnsupportedSchedule {
                    detail: format!(
                        "sycl backend v1: the scalar contiguous Elementwise schedule ONLY, got \
                         {other:?}"
                    ),
                });
            }
        }
        refuse_unsupported_nodes(plan.body)?;
        emit_scalar(plan)
    }
}

/// Refuse any body node whose lowering seam this backend has not been given, up front and as a
/// typed error: the builder's defaults for those seams panic, and a lowering must never unwind.
fn refuse_unsupported_nodes(body: &ScalarExpr) -> Result<(), LowerError> {
    let dag = ExprDag::from_expr(body);
    for id in 0..dag.len() as u32 {
        let node = dag.node(id);
        let ok = matches!(
            node,
            DagNode::Input(_)
                | DagNode::Const(_)
                | DagNode::Param(_)
                | DagNode::Add(..)
                | DagNode::Sub(..)
                | DagNode::Mul(..)
                | DagNode::Div(..)
                | DagNode::Unary(..)
                | DagNode::Binary(..)
                | DagNode::Select(..)
                | DagNode::Fma(..)
        );
        if !ok {
            return Err(LowerError::UnsupportedOp {
                detail: format!("sycl backend v1 has no lowering for body node {node:?}"),
            });
        }
    }
    Ok(())
}

fn emit_scalar(plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
    let ctype = scalar_ctype(plan.dtype).expect("supports_dtype gate guarantees a scalar ctype");
    let octype = out_ctype_of(plan, 0, ctype);
    let name = format!("bonefish_sycl_{}_{}", plan.op_name, dtype_tag(plan.dtype));
    let dtype = plan.dtype;

    let acc = |idx: u8| Ok(Spelling::Spelled(format!("in{idx}[i]")));
    let un = |op, x: String| sycl_unary(op, x, dtype);
    let bin = |op, a: String, b: String| sycl_binary(op, a, b, dtype);
    let sel = |c, a, b| sycl_select(c, a, b, dtype);
    let fm = |a, b, c| sycl_fma(a, b, c, dtype);
    // Constants stay `const_lit`'s double literals, as in `unpopped-cpu-c`: an f32 expression with
    // a constant evaluates in double and rounds once at the store, which is what the f64 oracle
    // does. Measured: spelling it as an f32 literal (`0.1f`) differs from the oracle by 1 ulp
    // (`f32_constant_matches_the_oracle`), so it was tried and reverted.
    let cst = |v| Ok(Spelling::Spelled(const_lit(v)));
    let lowering = Lowering::builder(&acc, &un, &bin)
        .select(&sel)
        .fma(&fm)
        .compute_dtype(dtype)
        .constant(&cst)
        .build();
    let dag = ExprDag::from_expr(plan.body);
    let (prelude, root) = lower_dag(&dag, ctype, &lowering)?;
    let store = store_expr_of(plan, 0, root);

    let mut s = String::new();
    s.push_str("// Generated by bonefish (sycl backend) — do not edit.\n");
    s.push_str(&format!(
        "// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    ));
    s.push_str("#include <sycl/sycl.hpp>\n#include <cmath>\n\n");
    s.push_str("#if defined(_WIN32)\n#define BONEFISH_EXPORT __declspec(dllexport)\n#else\n");
    s.push_str("#define BONEFISH_EXPORT __attribute__((visibility(\"default\")))\n#endif\n\n");
    s.push_str(
        "// Returns 0 ok, 1 SYCL exception, 2 other exception, 3 device lacks fp64.\n\
         // `queue` is a sycl::queue*; every pointer must be USM reachable from it.\n",
    );
    s.push_str(&format!(
        "extern \"C\" BONEFISH_EXPORT int {name}(void* queue,\n"
    ));
    for i in 0..plan.n_inputs {
        s.push_str(&format!("    const {ctype}* in{i},\n"));
    }
    s.push_str(&format!("    {octype}* out,\n"));
    s.push_str(&format!(
        "    long long n{}) {{\n",
        param_args(plan.body, param_ctype(plan))
    ));
    s.push_str("  try {\n");
    s.push_str("    sycl::queue& q = *static_cast<sycl::queue*>(queue);\n");
    if plan.dtype == ElementKind::F64 {
        s.push_str("    if (!q.get_device().has(sycl::aspect::fp64)) return 3;\n");
    }
    s.push_str("    if (n <= 0) return 0;\n");
    s.push_str(
        "    q.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> id) {\n",
    );
    s.push_str("      const long long i = static_cast<long long>(id[0]);\n");
    for decl in &prelude {
        s.push_str(&format!("      {decl}\n"));
    }
    s.push_str(&format!("      out[i] = {store};\n"));
    s.push_str("    }).wait();\n    return 0;\n");
    s.push_str("  } catch (const sycl::exception&) { return 1; } catch (...) { return 2; }\n}\n");
    Ok(GeneratedKernel::new(name, s))
}

fn declined_unary(op: UnaryOp, dtype: ElementKind) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
        op: DeclinedOp::Unary(op),
        dtype,
        why: "no lowering at this dtype in the sycl backend v1".to_string(),
    }))
}

/// The libm functions the neutral C spellers emit (`expf`, `fabs`, ...), each of which has a
/// same-named overload in the `sycl::` namespace.
const MATH: &[&str] = &[
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "cbrt",
    "ceil",
    "copysign",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "floor",
    "fma",
    "fmax",
    "fmin",
    "fmod",
    "lgamma",
    "log",
    "log1p",
    "log10",
    "log2",
    "nextafter",
    "pow",
    "rint",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
];

/// Rewrite every libm call in a neutral C spelling to its `sycl::` overload, or return the name
/// of a call that has no SYCL spelling here.
///
/// Why not leave `expf` alone: measured with icpx 2026.1 on the CPU device, the default
/// (fast) floating-point model compiles `expf(3.0f)` to 20.08906 where the correct value is
/// 20.08554, a relative error of 1.8e-4; `sycl::exp` is accurate under the same flags, and
/// `-ffp-model=precise` fixes `expf` but cannot travel with generated source. Refuse rather than
/// approximate: a call this table does not know is returned as `Err`, never emitted.
fn sycl_math(spelled: &str) -> Result<String, String> {
    let bytes = spelled.as_bytes();
    let mut out = String::with_capacity(spelled.len() + 16);
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if !(c.is_ascii_alphabetic() || c == '_') {
            out.push(c);
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let ident = &spelled[start..i];
        let qualified = start >= 2 && &spelled[start - 2..start] == "::";
        let is_call = bytes.get(i) == Some(&b'(');
        if !is_call || qualified {
            out.push_str(ident);
            continue;
        }
        let stem = if MATH.contains(&ident) {
            Some(ident)
        } else {
            ident.strip_suffix('f').filter(|s| MATH.contains(s))
        };
        match stem {
            Some(s) => {
                out.push_str("sycl::");
                out.push_str(s);
            }
            None => return Err(ident.to_string()),
        }
    }
    Ok(out)
}

/// A neutral spelling, rewritten for SYCL, or the typed decline naming the call it cannot spell.
fn spelled(s: String, op: DeclinedOp, dtype: ElementKind) -> Result<Spelling, LowerError> {
    match sycl_math(&s) {
        Ok(s) => Ok(Spelling::Spelled(s)),
        Err(call) => Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op,
            dtype,
            why: format!(
                "the neutral spelling calls `{call}`, which has no SYCL spelling in the sycl backend v1"
            ),
        })),
    }
}

fn sycl_unary(op: UnaryOp, x: String, dtype: ElementKind) -> Result<Spelling, LowerError> {
    match dtype {
        ElementKind::F32 => spelled(unary_f32(op, x), DeclinedOp::Unary(op), dtype),
        ElementKind::F64 => spelled(unary_f64(op, x), DeclinedOp::Unary(op), dtype),
        _ => declined_unary(op, dtype),
    }
}

fn sycl_binary(
    op: BinaryOp,
    a: String,
    b: String,
    dtype: ElementKind,
) -> Result<Spelling, LowerError> {
    match dtype {
        ElementKind::F32 => spelled(binary_f32(op, a, b), DeclinedOp::Binary(op), dtype),
        ElementKind::F64 => spelled(binary_f64(op, a, b), DeclinedOp::Binary(op), dtype),
        _ => Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Binary(op),
            dtype,
            why: "no lowering at this dtype in the sycl backend v1".to_string(),
        })),
    }
}

fn sycl_select(
    c: String,
    a: String,
    b: String,
    dtype: ElementKind,
) -> Result<Spelling, LowerError> {
    match dtype {
        ElementKind::F32 => spelled(select_f32(c, a, b), DeclinedOp::Select, dtype),
        ElementKind::F64 => spelled(select_f64(c, a, b), DeclinedOp::Select, dtype),
        _ => Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Select,
            dtype,
            why: "no lowering at this dtype in the sycl backend v1".to_string(),
        })),
    }
}

fn sycl_fma(a: String, b: String, c: String, dtype: ElementKind) -> Result<Spelling, LowerError> {
    match dtype {
        ElementKind::F32 | ElementKind::F64 => {
            spelled(format!("fma({a}, {b}, {c})"), DeclinedOp::Fma, dtype)
        }
        _ => Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Fma,
            dtype,
            why: "Fma is f32/f64 only in the sycl backend v1".to_string(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn libm_calls_become_sycl_overloads() {
        assert_eq!(sycl_math("expf(in0[i])").unwrap(), "sycl::exp(in0[i])");
        assert_eq!(
            sycl_math("fabs(sqrtf(x))").unwrap(),
            "sycl::fabs(sycl::sqrt(x))"
        );
        assert_eq!(sycl_math("(a + b) * in1[i]").unwrap(), "(a + b) * in1[i]");
    }

    #[test]
    fn rewriting_is_idempotent() {
        let once = sycl_math("powf(expf(a), log2f(b))").unwrap();
        assert_eq!(once, "sycl::pow(sycl::exp(a), sycl::log2(b))");
        assert_eq!(sycl_math(&once).unwrap(), once);
    }

    #[test]
    fn a_call_with_no_sycl_spelling_is_refused_by_name() {
        // Positive control: the neighbouring known call in the same string is fine on its own.
        assert!(sycl_math("expf(x)").is_ok());
        assert_eq!(sycl_math("expf(heaviside(x))").unwrap_err(), "heaviside");
        // An f-suffixed name that is not libm must not be mistaken for one.
        assert_eq!(sycl_math("stuff(x)").unwrap_err(), "stuff");
    }

    #[test]
    fn identifiers_that_are_not_calls_are_left_alone() {
        assert_eq!(sycl_math("exp + in0[i] * p0").unwrap(), "exp + in0[i] * p0");
    }
}
