//! `bonefish-bench` tests. They check that rows are well formed, that verification passes on the
//! real answer and FAILS on a deliberately wrong layout (with a margin, so the tolerance is
//! calibrated against a sabotage rather than assumed), and that the command line parses. They
//! never assert a speed. Device tests need a SYCL device and skip loudly without oneAPI unless
//! `BONEFISH_REQUIRE_SYCL=1`. A variant the device refuses is reported NOT RUN and fails under
//! `BONEFISH_REQUIRE_HALF=1`. CPU (OpenCL) device only; GPU UNVERIFIED.

mod common;

use bonefish::SYCL_AVAILABLE;
use bonefish_bench::{Plan, Shape, Target, Variant, parse_args, pick_device, row_json, run_row};

fn target(test: &str) -> Option<Target> {
    if !SYCL_AVAILABLE {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but built without oneAPI"
        );
        eprintln!("SKIP {test}: built without oneAPI. Nothing ran on a device.");
        return None;
    }
    let (queue, info) = pick_device(None, "cpu").expect("a CPU SYCL device");
    Some(Target {
        queue,
        info,
        power_note: "not recorded".into(),
        command_line: "test".into(),
    })
}

fn plan(inject: bool) -> Plan {
    Plan {
        shapes: vec![],
        variants: vec![],
        warmups: 1,
        runs: 3,
        samples: 64,
        inject_wrong_shape: inject,
        seed: 7,
    }
}

/// `Some(row)` if the variant ran; `None` (loudly) if the device refused it.
fn try_row(t: &Target, sh: Shape, v: Variant, inject: bool) -> Option<bonefish_bench::Row> {
    match run_row(&t.queue, sh, v, &plan(inject)) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("NOT RUN {} {sh:?}: {e}", v.name());
            assert!(
                !std::env::var("BONEFISH_REQUIRE_HALF").is_ok_and(|x| x == "1"),
                "{}: BONEFISH_REQUIRE_HALF=1 but it did not run: {e}",
                v.name()
            );
            None
        }
    }
}

const SHAPES: [Shape; 3] = [
    Shape {
        m: 1,
        k: 96,
        n: 40,
        batch: 1,
    },
    Shape {
        m: 7,
        k: 96,
        n: 40,
        batch: 1,
    },
    Shape {
        m: 5,
        k: 48,
        n: 24,
        batch: 3,
    },
];

#[test]
fn correct_results_verify_and_a_wrong_layout_does_not() {
    common::arm();
    common::guarded(|| {
        let Some(t) = target("verify") else { return };
        let mut ran = 0;
        for sh in SHAPES {
            for v in Variant::ALL {
                let Some(good) = try_row(&t, sh, v, false) else {
                    continue;
                };
                ran += 1;
                let g = &good.verification;
                assert!(g.passed, "{} {sh:?}: {g:?}", v.name());
                assert!(!g.injected_wrong_shape && g.entries > 0);
                // Sabotage: the same data checked against a reference with `w` read transposed.
                let bad = try_row(&t, sh, v, true).expect("ran a moment ago");
                let b = &bad.verification;
                assert!(!b.passed, "wrong layout passed: {} {sh:?}: {b:?}", v.name());
                // Calibration: the sabotage misses the tolerance by more than 10x, so the
                // tolerance is not so loose that a wrong layout hides inside it.
                assert!(
                    b.max_rel_err > 10.0 * b.tolerance,
                    "tolerance too loose for {} {sh:?}: {b:?}",
                    v.name()
                );
                // And the real error sits well under the tolerance (not scraping it).
                assert!(
                    g.max_rel_err * 2.0 < g.tolerance,
                    "{} {sh:?}: {g:?}",
                    v.name()
                );
            }
        }
        // Positive control: f32 and f64 at least ran everywhere (only half may be refused).
        assert!(ran >= 2 * SHAPES.len(), "only {ran} rows ran");
    });
}

#[test]
fn a_row_is_well_formed_json_with_its_predicate() {
    common::arm();
    common::guarded(|| {
        let Some(t) = target("json") else { return };
        let row = try_row(&t, SHAPES[2], Variant::F32, false).expect("f32 runs");
        let j = row_json(&t, &row);
        assert!(
            j.starts_with('{') && j.ends_with('}') && !j.contains('\n'),
            "{j}"
        );
        // Braces balance outside of strings and every quote is closed.
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        for c in j.chars() {
            if in_str {
                match (esc, c) {
                    (true, _) => esc = false,
                    (false, '\\') => esc = true,
                    (false, '"') => in_str = false,
                    _ => {}
                }
            } else {
                match c {
                    '"' => in_str = true,
                    '{' | '[' => depth += 1,
                    '}' | ']' => depth -= 1,
                    _ => {}
                }
                assert!(depth >= 0, "{j}");
            }
        }
        assert!(!in_str && depth == 0, "unbalanced: {j}");
        for key in [
            "\"op\":\"gemm_batch_strided\"",
            "\"batch\":3",
            "\"warmups\":1",
            "\"runs\":3",
            "\"timing_source\":",
            "\"median_ms\":",
            "\"min_ms\":",
            "\"max_ms\":",
            "\"launch_floor_median_ms\":",
            "\"gflops\":",
            "\"entries_checked\":64",
            "\"max_abs_err\":",
            "\"tolerance_rel\":",
            "\"driver_version\":",
            "\"command_line\":\"test\"",
            "\"os\":",
            "\"power_clock_state\":\"not recorded\"",
        ] {
            assert!(j.contains(key), "missing {key} in {j}");
        }
        // Order of the timings is a fact about the data, not a speed claim.
        let tm = &row.timing;
        assert!(
            tm.min_ms <= tm.median_ms && tm.median_ms <= tm.max_ms,
            "{tm:?}"
        );
        // M == 1 reports GB/s; M > 1 does not.
        let mv = try_row(&t, SHAPES[0], Variant::F32, false).expect("f32 runs");
        assert!(mv.gbps.is_some() && row.gbps.is_none());
    });
}

#[test]
fn the_command_line_parses_and_rejects() {
    common::arm();
    let a = |s: &[&str]| s.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
    let p = parse_args(&a(&[
        "--shape",
        "2,3,4",
        "--shape",
        "2,3,4,5",
        "--variant",
        "bgemm_f32",
        "--runs",
        "9",
        "--warmups",
        "2",
        "--samples",
        "11",
        "--inject-wrong-shape",
        "--device-index",
        "1",
    ]))
    .unwrap();
    assert_eq!(
        p.plan.shapes,
        [
            Shape {
                m: 2,
                k: 3,
                n: 4,
                batch: 1
            },
            Shape {
                m: 2,
                k: 3,
                n: 4,
                batch: 5
            }
        ]
    );
    assert_eq!(p.plan.variants, [Variant::BgemmF32]);
    assert_eq!((p.plan.runs, p.plan.warmups, p.plan.samples), (9, 2, 11));
    assert!(p.plan.inject_wrong_shape && p.index == Some(1));
    // Defaults: the smoke preset and every variant.
    let d = parse_args(&[]).unwrap();
    assert_eq!(d.plan.variants.len(), Variant::ALL.len());
    assert!(!d.plan.shapes.is_empty() && d.plan.runs == 10 && d.plan.warmups == 3);
    // The Fuel preset is the shapes row 127 names.
    let f = parse_args(&a(&["--preset", "fuel"])).unwrap();
    assert!(f.plan.shapes.contains(&Shape {
        m: 1,
        k: 5120,
        n: 5120,
        batch: 1
    }));
    assert!(f.plan.shapes.contains(&Shape {
        m: 1024,
        k: 5120,
        n: 17408,
        batch: 1
    }));
    // Rejections (each with a positive control above: the same flag, valid, parses).
    for bad in [
        &["--shape", "1,2"][..],
        &["--shape", "a,b,c"],
        &["--variant", "nope"],
        &["--runs", "0"],
        &["--runs"],
        &["--bogus"],
        &["--preset", "huge"],
    ] {
        assert!(parse_args(&a(bad)).is_err(), "{bad:?} was accepted");
    }
}

#[test]
fn help_and_version_succeed_without_a_device() {
    // CLI basics (OPS-011): both are successes, and neither needs a SYCL device.
    for flag in ["--help", "-h", "--version", "-V"] {
        bonefish_bench::cli(&[flag.to_string()]).unwrap_or_else(|e| panic!("{flag} failed: {e}"));
    }
}
