//! Command line and JSON output.

use std::fmt::Write as _;

use bonefish::Queue;

use crate::run::{Plan, Res, Row, Shape, Target, Variant, e, pick_device, run_row};

fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn jf(x: f64) -> String {
    if x.is_finite() {
        format!("{x:e}")
    } else {
        "null".into()
    }
}

fn jopt(s: Option<&str>) -> String {
    s.map_or("null".into(), js)
}

/// One JSON object (no trailing newline) for `row`, with the run's context.
#[must_use]
pub fn row_json(t: &Target, row: &Row) -> String {
    let (din, dout, acc) = row.variant.dtypes();
    let v = &row.verification;
    let tm = &row.timing;
    let dev = match t.info.as_ref() {
        Some(d) => format!(
            "{{\"index\":{},\"name\":{},\"vendor\":{},\"backend\":{},\"type\":{},\"driver_version\":{},\"pci_address\":{},\"uuid\":{}}}",
            d.index,
            js(d.name.trim()),
            js(&d.vendor),
            js(&d.backend),
            js(&format!("{:?}", d.device_type)),
            js(&d.driver_version),
            jopt(d.pci_address.as_deref()),
            jopt(d.uuid.as_deref()),
        ),
        None => "null".into(),
    };
    let sh = row.shape;
    format!(
        "{{\"tool\":\"bonefish-bench\",\"version\":{},\"op\":{},\"shape\":{{\"m\":{},\"k\":{},\"n\":{},\"batch\":{},\"layout\":\"row-major x[M,K] @ W[K,N] -> y[M,N], NoTrans/NoTrans, alpha 1, beta 0\"}},\"variant\":{},\"dtype_in\":{},\"dtype_out\":{},\"accumulate\":{},\"warmups\":{},\"runs\":{},\"timing_source\":{},\"median_ms\":{},\"min_ms\":{},\"max_ms\":{},\"launch_floor_median_ms\":{},\"gflops\":{},\"gbps\":{},\"verification\":{{\"entries_checked\":{},\"sampled\":true,\"max_abs_err\":{},\"max_rel_err\":{},\"tolerance_rel\":{},\"passed\":{},\"injected_wrong_shape\":{}}},\"device\":{},\"os\":{},\"arch\":{},\"build\":{},\"power_clock_state\":{},\"command_line\":{}}}",
        js(env!("CARGO_PKG_VERSION")),
        js(if sh.batch == 1 {
            "gemm"
        } else {
            "gemm_batch_strided"
        }),
        sh.m,
        sh.k,
        sh.n,
        sh.batch,
        js(row.variant.name()),
        js(din),
        js(dout),
        js(acc),
        row.warmups,
        row.runs,
        js(tm.source),
        jf(tm.median_ms),
        jf(tm.min_ms),
        jf(tm.max_ms),
        jf(tm.floor_median_ms),
        jf(row.gflops),
        row.gbps.map_or("null".into(), jf),
        v.entries,
        jf(v.max_abs_err),
        jf(v.max_rel_err),
        jf(v.tolerance),
        v.passed,
        v.injected_wrong_shape,
        dev,
        js(std::env::consts::OS),
        js(std::env::consts::ARCH),
        js(if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }),
        js(&t.power_note),
        js(&t.command_line),
    )
}

const USAGE: &str = "usage: bonefish-bench [--list-devices] [--kind cpu|gpu|any] [--device-index I]\n\
  [--shape M,K,N[,BATCH]]... [--preset smoke|fuel] [--variant NAME]... [--warmups W] [--runs N]\n\
  [--samples S] [--seed X] [--inject-wrong-shape] [--power-note TEXT]\n\
variants: f32 f64 hgemm hgemm_f32 bgemm bgemm_f32 (default: all)\n\
presets: smoke (small, default) | fuel (Fuel row 127: M=1/128/1024, K=N=5120, and K=5120,N=17408)\n\
Prints one JSON object per line. Defaults: --warmups 3 --runs 10 --samples 256.";

/// Parsed command line.
pub struct Parsed {
    /// What to run.
    pub plan: Plan,
    /// `cpu`, `gpu` or `any`.
    pub kind: String,
    /// Explicit device index.
    pub index: Option<usize>,
    /// Only list devices.
    pub list_devices: bool,
    /// Power/clock note to record.
    pub power_note: String,
}

/// Parse `args` (without the program name).
///
/// # Errors
///
/// Usage text for a bad argument.
pub fn parse_args(args: &[String]) -> Res<Parsed> {
    let mut p = Parsed {
        plan: Plan {
            shapes: vec![],
            variants: vec![],
            warmups: 3,
            runs: 10,
            samples: 256,
            inject_wrong_shape: false,
            seed: 0x00B0_4EF1_5400_0001,
        },
        kind: "cpu".into(),
        index: None,
        list_devices: false,
        power_note: "not recorded".into(),
    };
    let mut it = args.iter();
    let mut preset: Option<String> = None;
    while let Some(a) = it.next() {
        let mut val = |name: &str| -> Res<String> {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--list-devices" => p.list_devices = true,
            "--kind" => p.kind = val("--kind")?,
            "--device-index" => p.index = Some(val("--device-index")?.parse().map_err(e)?),
            "--shape" => {
                let s = val("--shape")?;
                let n: Vec<usize> = s
                    .split(',')
                    .map(|x| {
                        x.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad shape {s:?}"))
                    })
                    .collect::<Res<_>>()?;
                match n[..] {
                    [m, k, nn] => p.plan.shapes.push(Shape {
                        m,
                        k,
                        n: nn,
                        batch: 1,
                    }),
                    [m, k, nn, batch] => p.plan.shapes.push(Shape { m, k, n: nn, batch }),
                    _ => return Err(format!("shape must be M,K,N or M,K,N,BATCH, got {s:?}")),
                }
            }
            "--preset" => preset = Some(val("--preset")?),
            "--variant" => p.plan.variants.push(Variant::parse(&val("--variant")?)?),
            "--warmups" => p.plan.warmups = val("--warmups")?.parse().map_err(e)?,
            "--runs" => p.plan.runs = val("--runs")?.parse().map_err(e)?,
            "--samples" => p.plan.samples = val("--samples")?.parse().map_err(e)?,
            "--seed" => p.plan.seed = val("--seed")?.parse().map_err(e)?,
            "--inject-wrong-shape" => p.plan.inject_wrong_shape = true,
            "--power-note" => p.power_note = val("--power-note")?,
            "--help" | "-h" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    if p.plan.shapes.is_empty() {
        let sh = |m, k, n| Shape { m, k, n, batch: 1 };
        p.plan.shapes = match preset.as_deref().unwrap_or("smoke") {
            "smoke" => vec![sh(1, 256, 256), sh(64, 256, 256)],
            "fuel" => vec![
                sh(1, 5120, 5120),
                sh(128, 5120, 5120),
                sh(1024, 5120, 5120),
                sh(1024, 5120, 17408),
            ],
            other => return Err(format!("unknown preset {other:?}\n{USAGE}")),
        };
    }
    if p.plan.variants.is_empty() {
        p.plan.variants = Variant::ALL.to_vec();
    }
    if p.plan.runs == 0 {
        return Err("--runs must be at least 1".into());
    }
    Ok(p)
}

/// The whole CLI: JSON rows to stdout, notes to stderr. Fails if no row ran or any row failed
/// verification (its row is still printed, with `passed:false`).
///
/// # Errors
///
/// Usage errors, no device, no row ran, or a row failed verification.
pub fn cli(args: &[String]) -> Res<()> {
    // Help and version are successes: the text goes to stdout and the exit status is 0.
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("bonefish-bench {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let p = parse_args(args)?;
    if !bonefish::SYCL_AVAILABLE {
        return Err("built without oneAPI (SYCL_AVAILABLE is false): nothing can be timed".into());
    }
    if p.list_devices {
        for d in Queue::devices().map_err(e)? {
            println!(
                "{}: {} | {:?} | backend {} | driver {} | pci {:?} | uuid {:?}",
                d.index,
                d.name.trim(),
                d.device_type,
                d.backend,
                d.driver_version,
                d.pci_address,
                d.uuid
            );
        }
        return Ok(());
    }
    let (queue, info) = pick_device(p.index, &p.kind)?;
    let command_line = std::env::args().collect::<Vec<_>>().join(" ");
    let target = Target {
        queue,
        info,
        power_note: p.power_note.clone(),
        command_line,
    };
    if cfg!(debug_assertions) {
        eprintln!("bonefish-bench: debug build; numbers are not representative, use --release");
    }
    let (mut ran, mut failed) = (0usize, 0usize);
    for sh in &p.plan.shapes {
        for v in &p.plan.variants {
            match run_row(&target.queue, *sh, *v, &p.plan) {
                Ok(row) => {
                    println!("{}", row_json(&target, &row));
                    ran += 1;
                    failed += usize::from(!row.verification.passed);
                }
                Err(err) => eprintln!(
                    "bonefish-bench: {} {}x{}x{}: NOT RUN: {err}",
                    v.name(),
                    sh.m,
                    sh.k,
                    sh.n
                ),
            }
        }
    }
    if ran == 0 {
        return Err("no row ran".into());
    }
    if failed > 0 {
        return Err(format!(
            "{failed} row(s) failed verification (printed with passed:false)"
        ));
    }
    Ok(())
}
