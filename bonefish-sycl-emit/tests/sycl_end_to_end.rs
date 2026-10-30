//! Generate -> compile with `icpx -fsycl` -> run on a SYCL device -> compare to the oracle.
//!
//! Modelled on `unpopped-cpu-c`'s `cpu_end_to_end.rs`: the output is NaN-filled so a kernel that
//! launches and never writes is reported as NEVER-WROTE rather than as a wrong number, and the
//! reference is `unpopped::oracle`, which shares no lowering code with the emitter.
//!
//! Needs Intel's DPC++ compiler. Without it these tests skip loudly, *unless*
//! `BONEFISH_REQUIRE_SYCL=1`, which the oneAPI CI job sets so a missing toolchain fails instead
//! of passing vacuously. The decline tests at the bottom need no toolchain and always run.
//! Runs on whatever device the SYCL runtime offers as a CPU; GPU runs are UNVERIFIED.

use std::path::{Path, PathBuf};
use std::process::Command;

use bonefish_sycl_emit::Sycl;
use unpopped::backend::LowerError;
use unpopped::ir::{OpDef, input, konst};
use unpopped::oracle::{Fidelity, TypedBuffer, compare, evaluate};
use unpopped::{build_plan, generate, try_generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn find_icpx() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "icpx.exe" } else { "icpx" };
    if let Ok(p) = std::env::var("BONEFISH_ICPX") {
        return Some(p.into());
    }
    let mut roots: Vec<PathBuf> = std::env::var("ONEAPI_ROOT")
        .map(|r| vec![r.into()])
        .unwrap_or_default();
    roots.push(if cfg!(windows) {
        r"C:\Program Files (x86)\Intel\oneAPI".into()
    } else {
        "/opt/intel/oneapi".into()
    });
    for root in roots {
        let p = root.join("compiler").join("latest").join("bin").join(exe);
        if p.is_file() {
            return Some(p);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(exe))
            .find(|p| p.is_file())
    })
}

/// The compiler, or `None` after a loud skip. With `BONEFISH_REQUIRE_SYCL=1` a missing compiler
/// is a failure.
fn toolchain(test: &str) -> Option<PathBuf> {
    let found = find_icpx();
    if found.is_none() {
        assert!(
            std::env::var("BONEFISH_REQUIRE_SYCL").map_or(true, |v| v != "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but icpx was not found"
        );
        eprintln!("SKIP {test}: no icpx (set BONEFISH_ICPX or ONEAPI_ROOT). Nothing was compiled.");
    }
    found
}

fn lit(v: f64, ty: &str) -> String {
    if v.is_nan() {
        format!("({ty})NAN")
    } else if v.is_infinite() {
        format!("{}({ty})INFINITY", if v < 0.0 { "-" } else { "" })
    } else {
        format!("({ty}){v:?}")
    }
}

/// Wraps the emitted kernel in a `main` that fills inputs, NaN-fills the output, calls the kernel
/// on a CPU queue and prints the status and every output element.
fn harness(src: &str, name: &str, ty: &str, inputs: &[Vec<f64>]) -> String {
    let n = inputs[0].len();
    let mut s = String::new();
    s.push_str(src);
    s.push_str("\n#include <cstdio>\n#include <cmath>\n\nint main() {\n");
    s.push_str("  sycl::queue q{sycl::cpu_selector_v};\n");
    s.push_str(&format!(
        "  {ty}* out = sycl::malloc_shared<{ty}>({n}, q);\n"
    ));
    for (i, data) in inputs.iter().enumerate() {
        s.push_str(&format!(
            "  {ty}* in{i} = sycl::malloc_shared<{ty}>({n}, q);\n"
        ));
        for (j, v) in data.iter().enumerate() {
            s.push_str(&format!("  in{i}[{j}] = {};\n", lit(*v, ty)));
        }
    }
    s.push_str(&format!(
        "  for (int i = 0; i < {n}; ++i) out[i] = ({ty})NAN;\n"
    ));
    let args: Vec<String> = (0..inputs.len()).map(|i| format!("in{i}")).collect();
    s.push_str(&format!(
        "  int rc = {name}(&q, {}, out, {n});\n  printf(\"rc=%d\\n\", rc);\n",
        args.join(", ")
    ));
    s.push_str(&format!(
        "  for (int i = 0; i < {n}; ++i) printf(\"%.17g\\n\", (double)out[i]);\n  return 0;\n}}\n"
    ));
    s
}

fn parse_elem(line: &str) -> f64 {
    let t = line.trim();
    let l = t.to_ascii_lowercase();
    if l.starts_with("nan") || l.starts_with("-nan") || l.starts_with("+nan") {
        return f64::NAN;
    }
    t.parse::<f64>()
        .unwrap_or_else(|e| panic!("kernel printed an unparseable element {t:?}: {e}"))
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("bonefish-sycl-e2e")
        .join(format!("{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Variables for running the compiler and the program it builds without `setvars`: the
/// compiler's own `lib` (Intel's math runtime, `libmmd.lib`) for linking, and its `bin` for the
/// SYCL runtime. Prepended, so whatever the shell already set still applies.
fn oneapi_vars(icpx: &Path) -> Vec<(&'static str, std::ffi::OsString)> {
    let Some(root) = icpx.parent().and_then(Path::parent) else {
        return Vec::new();
    };
    let prepend = |var: &str, dir: PathBuf| {
        let mut parts = vec![dir];
        if let Some(old) = std::env::var_os(var) {
            parts.extend(std::env::split_paths(&old));
        }
        std::env::join_paths(parts).expect("path list")
    };
    let mut vars = vec![("PATH", prepend("PATH", root.join("bin")))];
    // Not `LIB` on Windows: setting it would replace the MSVC library paths icpx finds itself
    // (measured: `msvcrt.lib` then goes missing). The link step gets `-L` instead.
    if !cfg!(windows) {
        vars.push((
            "LD_LIBRARY_PATH",
            prepend("LD_LIBRARY_PATH", root.join("lib")),
        ));
    }
    vars
}

fn compile_and_run(icpx: &Path, dir: &Path, src: &str) -> Result<(i32, Vec<f64>), String> {
    let vars = oneapi_vars(icpx);
    let cpp = dir.join("k.cpp");
    let exe = dir.join(if cfg!(windows) { "k.exe" } else { "k" });
    std::fs::write(&cpp, src).map_err(|e| format!("write: {e}"))?;
    let out = Command::new(icpx)
        .envs(vars.iter().map(|(k, v)| (*k, v)))
        .args(["-fsycl", "-O2", "-o"])
        .arg(&exe)
        .arg(&cpp)
        .args(icpx.parent().and_then(Path::parent).map(|r| {
            let mut flag = std::ffi::OsString::from("-L");
            flag.push(r.join("lib"));
            flag
        }))
        .output()
        .map_err(|e| format!("spawn icpx: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "icpx failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let run = Command::new(&exe)
        .envs(vars.iter().map(|(k, v)| (*k, v)))
        .output()
        .map_err(|e| format!("run: {e}"))?;
    if !run.status.success() {
        return Err(format!(
            "kernel exe exited {:?}\n{}",
            run.status.code(),
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&run.stdout);
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let rc_line = lines.next().ok_or("no output")?;
    let rc: i32 = rc_line
        .trim()
        .strip_prefix("rc=")
        .ok_or_else(|| format!("first line was {rc_line:?}, expected rc=N"))?
        .parse()
        .map_err(|e| format!("bad rc: {e}"))?;
    Ok((rc, lines.map(parse_elem).collect()))
}

/// Generates the kernel for `op`, runs it, returns (actual, oracle).
fn run_op(
    icpx: &Path,
    tag: &str,
    op: &OpDef,
    dtype: ElementKind,
    cat: OpCategory,
    inputs: &[Vec<f64>],
) -> Result<(Vec<f64>, TypedBuffer), String> {
    let n = inputs[0].len() as i64;
    let d = OperandDesc::new(1, &[n], &[1], dtype, 256);
    let mut operands = vec![d; inputs.len()];
    operands.push(d);
    let key = structure_key(cat, &operands, ArchSku::Sm89);
    let kernel = generate(op, &key, &Sycl);
    let ty = if dtype == ElementKind::F64 {
        "double"
    } else {
        "float"
    };
    let src = harness(&kernel.source, &kernel.name, ty, inputs);
    let (rc, actual) = compile_and_run(icpx, &scratch(tag), &src)?;
    if rc != 0 {
        return Err(format!("kernel returned status {rc}"));
    }
    let plan = build_plan(op, &key);
    let bufs: Vec<TypedBuffer> = inputs
        .iter()
        .map(|d| match dtype {
            ElementKind::F64 => TypedBuffer::from_f64(&[n], d),
            _ => TypedBuffer::from_f32(&[n], &d.iter().map(|v| *v as f32).collect::<Vec<_>>()),
        })
        .collect();
    let expected = evaluate(&plan, &operands, &bufs, &[])
        .into_iter()
        .next()
        .unwrap();
    Ok((actual, expected))
}

/// NEVER-WROTE (a NaN survived where the oracle has a number) is reported apart from WRONG-VALUE.
fn check(
    actual: &[f64],
    expected: &TypedBuffer,
    dtype: ElementKind,
    fid: Fidelity,
) -> Result<(), String> {
    let want = expected.to_f64_vec();
    if actual.len() != want.len() {
        return Err(format!(
            "printed {} values, expected {}",
            actual.len(),
            want.len()
        ));
    }
    let unwritten: Vec<usize> = (0..want.len())
        .filter(|&i| actual[i].is_nan() && !want[i].is_nan())
        .collect();
    if !unwritten.is_empty() {
        return Err(format!(
            "NEVER-WROTE: NaN sentinel survived at {unwritten:?}"
        ));
    }
    let n = [actual.len() as i64];
    let got = match dtype {
        ElementKind::F64 => TypedBuffer::from_f64(&n, actual),
        _ => TypedBuffer::from_f32(&n, &actual.iter().map(|v| *v as f32).collect::<Vec<_>>()),
    };
    compare(expected, &got, fid).map_err(|e| format!("WRONG-VALUE vs oracle: {e}"))
}

// Seven lanes, not four: a 4-element contiguous cell plans as Vectorized, which v1 declines.
fn a7() -> Vec<f64> {
    vec![3.0, -1.5, 0.0, 2.25, -7.0, 10.0, 0.5]
}
fn b7() -> Vec<f64> {
    vec![0.0, 1.5, 4.0, -2.25, 5.0, -10.0, 0.25]
}

#[test]
fn add_and_mul_f32_match_the_oracle_bit_for_bit() {
    let Some(icpx) = toolchain("add_and_mul_f32") else {
        return;
    };
    for (tag, op) in [
        (
            "add_f32",
            OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1)),
        ),
        (
            "mul_f32",
            OpDef::elementwise("mul", 2, &[ElementKind::F32], input(0) * input(1)),
        ),
    ] {
        let (actual, expected) = run_op(
            &icpx,
            tag,
            &op,
            ElementKind::F32,
            OpCategory::BinaryElementwise,
            &[a7(), b7()],
        )
        .unwrap_or_else(|e| panic!("{tag}: {e}"));
        check(&actual, &expected, ElementKind::F32, Fidelity::BitExact)
            .unwrap_or_else(|e| panic!("{tag}: {e}"));
    }
}

#[test]
fn f64_add_matches_the_oracle() {
    let Some(icpx) = toolchain("f64_add") else {
        return;
    };
    let op = OpDef::elementwise("add", 2, &[ElementKind::F64], input(0) + input(1));
    let (actual, expected) = run_op(
        &icpx,
        "add_f64",
        &op,
        ElementKind::F64,
        OpCategory::BinaryElementwise,
        &[a7(), b7()],
    )
    .expect("add_f64 end-to-end");
    check(&actual, &expected, ElementKind::F64, Fidelity::BitExact).expect("add_f64");
}

#[test]
fn unary_exp_f32_is_within_tolerance() {
    let Some(icpx) = toolchain("unary_exp_f32") else {
        return;
    };
    let op = OpDef::elementwise("expop", 1, &[ElementKind::F32], input(0).exp());
    let (actual, expected) = run_op(
        &icpx,
        "exp_f32",
        &op,
        ElementKind::F32,
        OpCategory::UnaryElementwise,
        &[a7()],
    )
    .expect("exp_f32 end-to-end");
    check(
        &actual,
        &expected,
        ElementKind::F32,
        Fidelity::Tolerant {
            rel: 1e-5,
            abs: 1e-6,
        },
    )
    .expect("exp_f32");
}

/// `Max` must propagate NaN (conformance N2) after icpx has optimised the source. Single-NaN
/// lanes discriminate (`fmax` gets lane 6, both-NaN, right); finite lanes are the positive
/// control that the kernel ran at all.
#[test]
fn max_propagates_nan_through_icpx() {
    let Some(icpx) = toolchain("max_nan") else {
        return;
    };
    let nan = f64::NAN;
    let a = vec![nan, 1.0, 2.0, 3.0, -1.0, 5.0, nan];
    let b = vec![1.0, nan, 3.0, 2.0, -2.0, 4.0, nan];
    let op = OpDef::elementwise("maxop", 2, &[ElementKind::F32], input(0).max(input(1)));
    let (actual, expected) = run_op(
        &icpx,
        "max_nan_f32",
        &op,
        ElementKind::F32,
        OpCategory::BinaryElementwise,
        &[a, b],
    )
    .expect("max_nan end-to-end");
    let want = expected.to_f64_vec();
    for lane in [0, 1, 6] {
        assert!(
            want[lane].is_nan(),
            "oracle no longer propagates NaN at lane {lane}: {want:?}"
        );
        assert!(
            actual[lane].is_nan(),
            "lane {lane} must be NaN, got {actual:?}"
        );
    }
    for (lane, v) in [(2, 3.0), (3, 3.0), (4, -1.0), (5, 5.0)] {
        assert_eq!(actual[lane], v, "finite lane {lane}: {actual:?}");
    }
}

#[test]
fn double_kernel_reports_status_not_a_crash_when_fp64_is_missing() {
    // The generated host function checks aspect::fp64 and returns 3. The CPU device has fp64, so
    // this asserts the check is *present* in the source (a device without fp64 cannot be faked
    // here); the run above proves it does not fire when fp64 exists.
    let op = OpDef::elementwise("add", 2, &[ElementKind::F64], input(0) + input(1));
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F64, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    let k = generate(&op, &key, &Sycl);
    assert!(k.source.contains("aspect::fp64"), "{}", k.source);
    let d32 = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 256);
    let op32 = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let key32 = structure_key(
        OpCategory::BinaryElementwise,
        &[d32, d32, d32],
        ArchSku::Sm89,
    );
    assert!(
        !generate(&op32, &key32, &Sycl)
            .source
            .contains("aspect::fp64")
    );
}

/// Negative control for the harness itself: an all-NaN output (a kernel that never wrote) must be
/// reported as NEVER-WROTE, and a wrong number as WRONG-VALUE. Needs no toolchain.
#[test]
fn the_checker_distinguishes_never_wrote_from_wrong_value() {
    let want = TypedBuffer::from_f32(&[3], &[1.0, 2.0, 3.0]);
    let e = check(&[f64::NAN; 3], &want, ElementKind::F32, Fidelity::BitExact).unwrap_err();
    assert!(e.starts_with("NEVER-WROTE"), "{e}");
    let e = check(
        &[1.0, 2.0, 4.0],
        &want,
        ElementKind::F32,
        Fidelity::BitExact,
    )
    .unwrap_err();
    assert!(e.starts_with("WRONG-VALUE"), "{e}");
    check(
        &[1.0, 2.0, 3.0],
        &want,
        ElementKind::F32,
        Fidelity::BitExact,
    )
    .unwrap();
}

// ---- declines: no toolchain needed, always run ------------------------------------------------

fn key_for(dt: ElementKind, n: i64) -> unpopped_vocab::StructureKey {
    let d = OperandDesc::new(1, &[n], &[1], dt, 256);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89)
}

#[test]
fn unsupported_dtype_is_a_typed_decline() {
    let op = OpDef::elementwise("add", 2, &[ElementKind::I32], input(0) + input(1));
    match try_generate(&op, &key_for(ElementKind::I32, 7), &Sycl) {
        Err(LowerError::UnsupportedDtype { dtype, .. }) => assert_eq!(dtype, ElementKind::I32),
        other => panic!("expected UnsupportedDtype, got {other:?}"),
    }
}

#[test]
fn non_scalar_schedule_is_a_typed_decline() {
    // A 4-element contiguous f32 cell plans as Vectorized (see a7() note); v1 refuses it. The
    // positive control is the n = 7 case in the same call shape, which lowers.
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    assert!(
        try_generate(&op, &key_for(ElementKind::F32, 7), &Sycl).is_ok(),
        "n = 7 must lower (positive control)"
    );
    match try_generate(&op, &key_for(ElementKind::F32, 4), &Sycl) {
        Err(LowerError::UnsupportedSchedule { .. }) => {}
        other => panic!("expected UnsupportedSchedule, got {other:?}"),
    }
}

#[test]
fn generated_source_names_its_abi() {
    let op = OpDef::elementwise(
        "axpy",
        2,
        &[ElementKind::F32],
        input(0) * konst(2.0) + input(1),
    );
    let k = generate(&op, &key_for(ElementKind::F32, 7), &Sycl);
    assert!(k.name.starts_with("bonefish_sycl_axpy_"), "{}", k.name);
    assert!(
        k.source.contains(&format!("int {}(void* queue,", k.name)),
        "{}",
        k.source
    );
    assert!(k.source.contains("parallel_for"));
}

/// A constant in an f32 expression follows the reference emitter and the oracle: it is a double
/// literal, so the expression evaluates in double and rounds once at the store. Spelling it as an
/// f32 literal was tried first and differs from the oracle by 1 ulp for this input, so this
/// bit-exact comparison pins the choice in either direction.
#[test]
fn f32_constant_matches_the_oracle() {
    let Some(icpx) = toolchain("f32_constant") else {
        return;
    };
    let op = OpDef::elementwise("scale", 1, &[ElementKind::F32], input(0) * konst(0.1));
    let x: Vec<f64> = (1..=7)
        .map(|i| i as f64 * 1.37)
        .map(|v| v as f32 as f64)
        .collect();
    let (actual, expected) = run_op(
        &icpx,
        "scale_f32",
        &op,
        ElementKind::F32,
        OpCategory::UnaryElementwise,
        &[x],
    )
    .expect("scale_f32 end-to-end");
    check(&actual, &expected, ElementKind::F32, Fidelity::BitExact).expect("scale_f32");
}

/// Toolchain-free: no generated source may call a bare libm function (`expf`, `sqrt`, ...), which
/// icpx's default fast model compiles to a low-accuracy version. Positive control: the same
/// kernels do contain the `sycl::` spelling.
#[test]
fn generated_source_never_calls_bare_libm() {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89);
    for (name, e) in [
        ("exp", input(0).exp()),
        ("sqrt", input(0).sqrt()),
        ("tanh", input(0).tanh()),
    ] {
        let op = OpDef::elementwise(name, 1, &[ElementKind::F32], e);
        let src = generate(&op, &key, &Sycl).source;
        assert!(src.contains(&format!("sycl::{name}(")), "{name}: {src}");
        for bare in [
            format!("{name}f("),
            format!(" {name}("),
            format!("({name}("),
        ] {
            assert!(!src.contains(&bare), "{name}: bare `{bare}` in\n{src}");
        }
    }
}
