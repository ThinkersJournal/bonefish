//! The gate for the proposed `sycl:` vocabulary (KISS-CLASSIFY-6.8-0014): evaluates each token's
//! derivability witness against the architecture table that ships with the installed DPC++
//! compiler, and against the compiler itself.
//!
//! Needs an installed oneAPI. Without one these tests skip loudly, unless
//! `BONEFISH_REQUIRE_SYCL=1` (the oneAPI CI job), which turns a missing toolchain into a failure.
//! The mutation tests need only the table text and check that this gate can fail *for the reason
//! the clause names*: a token that no upstream entry produces.

use std::path::{Path, PathBuf};
use std::process::Command;

use bonefish_sycl_emit::target::{
    JIT_TOKEN, TOKENS, derive_tokens, parse_architecture_def, unwitnessed,
};

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

fn toolchain(test: &str) -> Option<PathBuf> {
    let found = find_icpx();
    if found.is_none() {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but icpx was not found"
        );
        eprintln!("SKIP {test}: no icpx. Nothing was checked.");
    }
    found
}

fn architecture_def(icpx: &Path) -> String {
    let root = icpx
        .parent()
        .and_then(Path::parent)
        .expect("icpx is in <root>/bin");
    let path = root
        .join("include")
        .join("sycl")
        .join("ext")
        .join("oneapi")
        .join("experimental")
        .join("device_architecture.def");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn every_named_token_is_produced_by_the_installed_table() {
    let Some(icpx) = toolchain("every_named_token_is_produced") else {
        return;
    };
    let table = parse_architecture_def(&architecture_def(&icpx));
    // Positive control: the parser found the table, and found aliases in it.
    assert!(
        table.len() > 50 && table.iter().any(|e| e.alias),
        "{}",
        table.len()
    );
    let missing = unwitnessed(TOKENS, &table);
    assert!(
        missing.is_empty(),
        "no upstream entry produces {missing:?}: the vocabulary names a capability the \
         installed DPC++ does not have"
    );
}

#[test]
fn the_vocabulary_is_complete_for_the_installed_table() {
    let Some(icpx) = toolchain("the_vocabulary_is_complete") else {
        return;
    };
    let table = parse_architecture_def(&architecture_def(&icpx));
    let unlisted: Vec<String> = derive_tokens(&table)
        .into_iter()
        .filter(|t| !TOKENS.contains(&t.as_str()))
        .collect();
    assert!(
        unlisted.is_empty(),
        "the installed DPC++ has architectures the vocabulary does not list: {unlisted:?}"
    );
}

/// Mutation: remove one upstream entry and the gate must fail, naming that token. This is the
/// reason the clause names, as opposed to an adjacent one such as "the file did not parse".
#[test]
fn the_gate_fails_for_the_reason_the_clause_names() {
    let Some(icpx) = toolchain("the_gate_fails_for_the_reason") else {
        return;
    };
    let text = architecture_def(&icpx);
    let mutated: String = text
        .lines()
        .filter(|l| !l.contains("__SYCL_ARCHITECTURE(intel_gpu_pvc,"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_ne!(mutated.len(), text.len(), "the mutation did not apply");
    let missing = unwitnessed(TOKENS, &parse_architecture_def(&mutated));
    assert_eq!(missing, ["sycl:intel-gpu-pvc"]);
    // And the unmutated table passes, so the failure above is the mutation's.
    assert!(unwitnessed(TOKENS, &parse_architecture_def(&text)).is_empty());
}

fn compiles_for(icpx: &Path, target: &str) -> bool {
    let dir = std::env::temp_dir().join(format!("bonefish-sycl-tokens-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let cpp = dir.join("t.cpp");
    std::fs::write(
        &cpp,
        "#include <sycl/sycl.hpp>\nint main() { sycl::queue q{sycl::cpu_selector_v}; return 0; }\n",
    )
    .expect("write");
    Command::new(icpx)
        .args(["-fsycl", &format!("-fsycl-targets={target}"), "-c", "-o"])
        .arg(dir.join(if cfg!(windows) { "t.obj" } else { "t.o" }))
        .arg(&cpp)
        .output()
        .expect("spawn icpx")
        .status
        .success()
}

/// The two token shapes the table cannot witness on its own: the portable JIT token, and that an
/// `intel_gpu_*` name is a target the compiler really accepts. Compile-only: no device is needed
/// (or used) to ask whether a target is valid. A made-up target is the negative control.
#[test]
fn the_compiler_accepts_the_named_targets() {
    let Some(icpx) = toolchain("the_compiler_accepts_the_named_targets") else {
        return;
    };
    assert!(
        !compiles_for(&icpx, "intel_gpu_acm_g10_not_a_target"),
        "control must fail"
    );
    assert!(compiles_for(
        &icpx,
        JIT_TOKEN.strip_prefix("sycl:").unwrap()
    ));
    assert!(compiles_for(&icpx, "intel_gpu_acm_g10"));
}
