//! Operands, timing and verification of one `(shape, variant)` row.

use std::time::Instant;

use bonefish::half::{bf16, f16};
use bonefish::{
    DeviceInfo, DeviceType, Element, Error, Event, Queue, Read, Scope, Transpose, UsmBuffer,
    UsmKind, Write,
};

/// Result type of the harness: errors are human-readable text.
pub type Res<T> = std::result::Result<T, String>;

pub(crate) fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}

/// The `gemm` variants the harness can time.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Variant {
    /// `f32` throughout.
    F32,
    /// `f64` throughout.
    F64,
    /// `f16` in, `f16` out (accumulation unstated by oneMKL).
    Hgemm,
    /// `f16` in, `f32` out and accumulate.
    HgemmF32,
    /// `bf16` in, `bf16` out, `f32` scalars.
    Bgemm,
    /// `bf16` in, `f32` out and accumulate.
    BgemmF32,
}

impl Variant {
    /// Every variant, in report order.
    pub const ALL: [Variant; 6] = [
        Variant::F32,
        Variant::F64,
        Variant::Hgemm,
        Variant::HgemmF32,
        Variant::Bgemm,
        Variant::BgemmF32,
    ];

    /// The name used on the command line and in the JSON.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Variant::F32 => "f32",
            Variant::F64 => "f64",
            Variant::Hgemm => "hgemm",
            Variant::HgemmF32 => "hgemm_f32",
            Variant::Bgemm => "bgemm",
            Variant::BgemmF32 => "bgemm_f32",
        }
    }

    /// Parse a command-line name.
    ///
    /// # Errors
    ///
    /// Text naming the valid choices.
    pub fn parse(s: &str) -> Res<Variant> {
        Variant::ALL
            .into_iter()
            .find(|v| v.name() == s)
            .ok_or_else(|| {
                let all: Vec<_> = Variant::ALL.iter().map(|v| v.name()).collect();
                format!("unknown variant {s:?}; choose from {}", all.join(", "))
            })
    }

    /// `(input dtype, output dtype, accumulate as far as it is stated)`.
    #[must_use]
    pub fn dtypes(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Variant::F32 => ("f32", "f32", "f32"),
            Variant::F64 => ("f64", "f64", "f64"),
            Variant::Hgemm => ("f16", "f16", "unstated by oneMKL"),
            Variant::HgemmF32 => ("f16", "f32", "f32"),
            Variant::Bgemm => ("bf16", "bf16", "f32 scalars; accumulation unstated"),
            Variant::BgemmF32 => ("bf16", "f32", "f32"),
        }
    }

    fn eps_out(self) -> f64 {
        match self {
            Variant::F32 | Variant::HgemmF32 | Variant::BgemmF32 => 2f64.powi(-24),
            Variant::F64 => 2f64.powi(-53),
            Variant::Hgemm => 2f64.powi(-11),
            Variant::Bgemm => 2f64.powi(-8),
        }
    }

    /// The accumulation unit used for the tolerance. The all-half `hgemm` states none, so the
    /// tolerance assumes the worst (a half accumulator).
    fn eps_acc(self) -> f64 {
        match self {
            Variant::F64 => 2f64.powi(-53),
            Variant::Hgemm => 2f64.powi(-11),
            _ => 2f64.powi(-24),
        }
    }

    fn in_bytes(self) -> usize {
        match self {
            Variant::F32 => 4,
            Variant::F64 => 8,
            _ => 2,
        }
    }

    fn out_bytes(self) -> usize {
        match self {
            Variant::F64 => 8,
            Variant::F32 | Variant::HgemmF32 | Variant::BgemmF32 => 4,
            Variant::Hgemm | Variant::Bgemm => 2,
        }
    }

    /// Relative tolerance, in units of `sum |a_ik * b_kj|`: a random-walk bound on the
    /// accumulation (`4 sqrt(K) eps_acc`) plus two roundings of the output. Calibrated by the
    /// sabotage test: a wrong layout misses it by far more than 10x on the sizes tested.
    #[must_use]
    pub fn tolerance(self, k: usize) -> f64 {
        4.0 * (k as f64).sqrt() * self.eps_acc() + 2.0 * self.eps_out()
    }
}

/// One shape: `y[m,n] = x[m,k] @ w[k,n]`, `batch` independent products.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Rows of `x` and `y`.
    pub m: usize,
    /// Columns of `x`, rows of `w`.
    pub k: usize,
    /// Columns of `w` and `y`.
    pub n: usize,
    /// Number of products (1 is a plain `gemm`).
    pub batch: usize,
}

/// What to run, declared up front.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Shapes to time.
    pub shapes: Vec<Shape>,
    /// Variants to time.
    pub variants: Vec<Variant>,
    /// Warm-up runs per row, not timed.
    pub warmups: usize,
    /// Timed runs per row.
    pub runs: usize,
    /// Output entries to verify per row.
    pub samples: usize,
    /// Verify against a reference that indexes `w` transposed, to show verification can fail.
    pub inject_wrong_shape: bool,
    /// Seed of the operand generator.
    pub seed: u64,
}

/// A device and queue to run on, with the context the rows carry.
pub struct Target {
    /// The queue (profiling enabled).
    pub queue: Queue,
    /// What the runtime says about the device.
    pub info: Option<DeviceInfo>,
    /// Free text from the command line (power/clock state is not read, only recorded).
    pub power_note: String,
    /// The exact command line.
    pub command_line: String,
}

/// Choose the device: `--device-index`, else the first of the requested type (`cpu` default).
///
/// # Errors
///
/// If SYCL is unavailable, or no such device exists.
pub fn pick_device(index: Option<usize>, kind: &str) -> Res<(Queue, Option<DeviceInfo>)> {
    let devices = Queue::devices().map_err(e)?;
    let chosen = match index {
        Some(i) => devices.iter().find(|d| d.index == i),
        None => devices.iter().find(|d| {
            matches!(
                (&d.device_type, kind),
                (DeviceType::Cpu, "cpu") | (DeviceType::Gpu, "gpu") | (_, "any")
            )
        }),
    }
    .ok_or_else(|| {
        let list: Vec<_> = devices
            .iter()
            .map(|d| {
                format!(
                    "{}: {} ({:?}, {})",
                    d.index,
                    d.name.trim(),
                    d.device_type,
                    d.backend
                )
            })
            .collect();
        format!(
            "no matching device; the runtime lists: [{}]",
            list.join("; ")
        )
    })?;
    let q = Queue::with_device_profiling(chosen).map_err(e)?;
    Ok((q, Some(chosen.clone())))
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Submit one row-major product on `s`: a column-major call with the operands swapped, so `a` is
/// `w` (`n x k` column-major, ld `n`) and `b` is `x` (`k x m`, ld `k`), `c` is `y` (`n x m`).
trait Submit<A: Element, C: Element> {
    fn go<'scope, 'env: 'scope>(
        s: &'scope Scope<'scope, 'env>,
        sh: Shape,
        a: &Read<'scope, A>,
        b: &Read<'scope, A>,
        c: &Write<'scope, C>,
    ) -> bonefish::Result<Event<'scope>>;
}

macro_rules! plain_impl {
    ($ty:ident, $t:ty) => {
        struct $ty;
        impl Submit<$t, $t> for $ty {
            fn go<'scope, 'env: 'scope>(
                s: &'scope Scope<'scope, 'env>,
                sh: Shape,
                a: &Read<'scope, $t>,
                b: &Read<'scope, $t>,
                c: &Write<'scope, $t>,
            ) -> bonefish::Result<Event<'scope>> {
                let (no, m, n, k) = (Transpose::No, sh.n, sh.m, sh.k);
                if sh.batch == 1 {
                    s.gemm(no, no, m, n, k, 1.0, a, sh.n, b, sh.k, 0.0, c, sh.n)
                } else {
                    s.gemm_batch_strided(
                        no,
                        no,
                        m,
                        n,
                        k,
                        1.0,
                        a,
                        sh.n,
                        sh.k * sh.n,
                        b,
                        sh.k,
                        sh.m * sh.k,
                        0.0,
                        c,
                        sh.n,
                        sh.m * sh.n,
                        sh.batch,
                    )
                }
            }
        }
    };
}
plain_impl!(F32, f32);
plain_impl!(F64, f64);

macro_rules! half_impl {
    ($ty:ident, $a:ty, $c:ty, $single:ident, $batched:ident) => {
        struct $ty;
        impl Submit<$a, $c> for $ty {
            fn go<'scope, 'env: 'scope>(
                s: &'scope Scope<'scope, 'env>,
                sh: Shape,
                a: &Read<'scope, $a>,
                b: &Read<'scope, $a>,
                c: &Write<'scope, $c>,
            ) -> bonefish::Result<Event<'scope>> {
                let (no, m, n, k) = (Transpose::No, sh.n, sh.m, sh.k);
                if sh.batch == 1 {
                    s.$single(no, no, m, n, k, 1.0, a, sh.n, b, sh.k, 0.0, c, sh.n)
                } else {
                    s.$batched(
                        no,
                        no,
                        m,
                        n,
                        k,
                        1.0,
                        a,
                        sh.n,
                        sh.k * sh.n,
                        b,
                        sh.k,
                        sh.m * sh.k,
                        0.0,
                        c,
                        sh.n,
                        sh.m * sh.n,
                        sh.batch,
                    )
                }
            }
        }
    };
}
half_impl!(Hg, f16, f16, hgemm, hgemm_batch_strided);
half_impl!(HgF32, f16, f32, hgemm_f32, hgemm_f32_batch_strided);
half_impl!(Bg, bf16, bf16, bgemm, bgemm_batch_strided);
half_impl!(BgF32, bf16, f32, bgemm_f32, bgemm_f32_batch_strided);

/// Verification result of one row.
#[derive(Clone, Debug)]
pub struct Verification {
    /// Entries checked.
    pub entries: usize,
    /// Largest absolute error against the f64 reference.
    pub max_abs_err: f64,
    /// Largest `|err| / sum|a*b|` (the quantity the tolerance bounds).
    pub max_rel_err: f64,
    /// The tolerance used, in the same units as `max_rel_err`.
    pub tolerance: f64,
    /// `max_rel_err <= tolerance` and every checked entry finite.
    pub passed: bool,
    /// The reference was deliberately given the wrong layout.
    pub injected_wrong_shape: bool,
}

/// Timing of one row.
#[derive(Clone, Debug)]
pub struct Timing {
    /// `event_profiling` or `wall_minus_floor`.
    pub source: &'static str,
    /// Median over the timed runs, in ms.
    pub median_ms: f64,
    /// Fastest timed run, in ms.
    pub min_ms: f64,
    /// Slowest timed run, in ms.
    pub max_ms: f64,
    /// Median of the empty-kernel launch floor, in ms (same source).
    pub floor_median_ms: f64,
}

/// One finished row.
#[derive(Clone, Debug)]
pub struct Row {
    /// The shape.
    pub shape: Shape,
    /// The variant.
    pub variant: Variant,
    /// Warm-ups declared.
    pub warmups: usize,
    /// Timed runs declared (and done).
    pub runs: usize,
    /// Timing.
    pub timing: Timing,
    /// GFLOP/s at the median, `2*M*N*K*batch / t`.
    pub gflops: f64,
    /// GB/s at the median, only for `M == 1` (the decode matvec).
    pub gbps: Option<f64>,
    /// Verification.
    pub verification: Verification,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Run `once` `warmups + runs` times and return the `runs` timings in ms and the source used.
/// `once` returns device ms, or `None` when the device cannot report time (wall time is used).
fn time_runs(
    warmups: usize,
    runs: usize,
    mut once: impl FnMut() -> Res<Option<f64>>,
) -> Res<(Vec<f64>, &'static str)> {
    let mut out = Vec::with_capacity(runs);
    let mut source = "event_profiling";
    for i in 0..warmups + runs {
        let t0 = Instant::now();
        let dev = once()?;
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let ms = dev.unwrap_or_else(|| {
            source = "wall_minus_floor";
            wall
        });
        if i >= warmups {
            out.push(ms);
        }
    }
    Ok((out, source))
}

fn device_ms(ev: &Event<'_>) -> Res<Option<f64>> {
    match ev.device_time() {
        Ok(t) if t.end_ns > t.start_ns => Ok(Some(t.duration_ns() as f64 / 1e6)),
        Ok(_) | Err(Error::InvalidArgument(_) | Error::Unavailable(_)) => Ok(None),
        Err(x) => Err(e(x)),
    }
}

/// The launch floor: median of `runs` empty-kernel submissions (after `warmups`), ms.
fn launch_floor(queue: &Queue, warmups: usize, runs: usize) -> Res<f64> {
    let (mut v, _) = time_runs(warmups, runs, || {
        queue
            .scope(|s| {
                let ev = s.noop()?;
                device_ms(&ev).map_err(Error::Other)
            })
            .map_err(e)
    })?;
    Ok(median(&mut v))
}

fn fill<A>(len: usize, rng: &mut Rng, conv: impl Fn(f64) -> A) -> Vec<A> {
    (0..len).map(|_| conv(rng.unit())).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_generic<A, C, S>(
    queue: &Queue,
    variant: Variant,
    sh: Shape,
    plan: &Plan,
    floor: f64,
    conv: impl Fn(f64) -> A,
    a2f: impl Fn(A) -> f64,
    c2f: impl Fn(C) -> f64,
) -> Res<Row>
where
    A: Element,
    C: Element,
    S: Submit<A, C>,
{
    let mut rng = Rng(plan.seed ^ ((sh.m * 31 + sh.k * 17 + sh.n * 7 + sh.batch) as u64));
    let xh = fill(sh.batch * sh.m * sh.k, &mut rng, &conv);
    let wh = fill(sh.batch * sh.k * sh.n, &mut rng, &conv);
    let x = UsmBuffer::from_slice(queue, UsmKind::Device, &xh).map_err(e)?;
    let w = UsmBuffer::from_slice(queue, UsmKind::Device, &wh).map_err(e)?;
    let mut y =
        UsmBuffer::<C>::zeroed(queue, UsmKind::Device, sh.batch * sh.m * sh.n).map_err(e)?;

    let (times, source) = time_runs(plan.warmups, plan.runs, || {
        queue
            .scope(|s| {
                let (a, b, c) = (s.read(&w)?, s.read(&x)?, s.write(&mut y)?);
                let ev = S::go(s, sh, &a, &b, &c)?;
                device_ms(&ev).map_err(Error::Other)
            })
            .map_err(e)
    })?;
    let mut sorted = times;
    let mut med = median(&mut sorted);
    let (mut min, mut max) = (sorted[0], sorted[sorted.len() - 1]);
    if source == "wall_minus_floor" {
        // Wall time includes the host path; the documented fallback subtracts the floor.
        (med, min, max) = (
            (med - floor).max(0.0),
            (min - floor).max(0.0),
            (max - floor).max(0.0),
        );
    }

    // Read back once, outside the timed region, and check a sample against f64.
    let yh = y.to_vec().map_err(e)?;
    let mut s_rng = Rng(plan.seed ^ 0xA5A5);
    let (mut max_abs, mut max_rel, mut finite) = (0.0_f64, 0.0_f64, true);
    let entries = plan.samples.min(sh.batch * sh.m * sh.n);
    for _ in 0..entries {
        let b = (s_rng.next_u64() % sh.batch as u64) as usize;
        let i = (s_rng.next_u64() % sh.m as u64) as usize;
        let j = (s_rng.next_u64() % sh.n as u64) as usize;
        let (mut dot, mut abs) = (0.0_f64, 0.0_f64);
        for l in 0..sh.k {
            let xv = a2f(xh[b * sh.m * sh.k + i * sh.k + l]);
            // Row-major w[k, n] is element l * n + j; the injected fault reads it transposed.
            let widx = if plan.inject_wrong_shape {
                j * sh.k + l
            } else {
                l * sh.n + j
            };
            let wv = a2f(wh[b * sh.k * sh.n + widx]);
            dot += xv * wv;
            abs += (xv * wv).abs();
        }
        let got = c2f(yh[b * sh.m * sh.n + i * sh.n + j]);
        finite &= got.is_finite();
        let err = (got - dot).abs();
        max_abs = max_abs.max(err);
        if abs > 0.0 {
            max_rel = max_rel.max(err / abs);
        }
    }
    let tol = variant.tolerance(sh.k);
    let flops = 2.0 * (sh.m * sh.n * sh.k * sh.batch) as f64;
    let secs = med / 1e3;
    let bytes =
        (sh.k * sh.n + sh.m * sh.k) * variant.in_bytes() + sh.m * sh.n * variant.out_bytes();
    Ok(Row {
        shape: sh,
        variant,
        warmups: plan.warmups,
        runs: plan.runs,
        timing: Timing {
            source,
            median_ms: med,
            min_ms: min,
            max_ms: max,
            floor_median_ms: floor,
        },
        gflops: if secs > 0.0 {
            flops / secs / 1e9
        } else {
            f64::NAN
        },
        gbps: (sh.m == 1 && secs > 0.0).then(|| (bytes * sh.batch) as f64 / secs / 1e9),
        verification: Verification {
            entries,
            max_abs_err: max_abs,
            max_rel_err: max_rel,
            tolerance: tol,
            passed: finite && max_rel <= tol,
            injected_wrong_shape: plan.inject_wrong_shape,
        },
    })
}

/// Time and verify one `(shape, variant)` row.
///
/// # Errors
///
/// Text describing the device or API error (a device that lacks a half variant is one).
pub fn run_row(queue: &Queue, sh: Shape, variant: Variant, plan: &Plan) -> Res<Row> {
    if sh.m == 0 || sh.k == 0 || sh.n == 0 || sh.batch == 0 || plan.runs == 0 {
        return Err("shape dimensions, batch and runs must be at least 1".into());
    }
    let fl = launch_floor(queue, plan.warmups, plan.runs)?;
    match variant {
        Variant::F32 => run_generic::<f32, f32, F32>(
            queue,
            variant,
            sh,
            plan,
            fl,
            |x| x as f32,
            f64::from,
            f64::from,
        ),
        Variant::F64 => {
            run_generic::<f64, f64, F64>(queue, variant, sh, plan, fl, |x| x, |x| x, |x| x)
        }
        Variant::Hgemm => run_generic::<f16, f16, Hg>(
            queue,
            variant,
            sh,
            plan,
            fl,
            f16::from_f64,
            f16::to_f64,
            f16::to_f64,
        ),
        Variant::HgemmF32 => run_generic::<f16, f32, HgF32>(
            queue,
            variant,
            sh,
            plan,
            fl,
            f16::from_f64,
            f16::to_f64,
            f64::from,
        ),
        Variant::Bgemm => run_generic::<bf16, bf16, Bg>(
            queue,
            variant,
            sh,
            plan,
            fl,
            bf16::from_f64,
            bf16::to_f64,
            bf16::to_f64,
        ),
        Variant::BgemmF32 => run_generic::<bf16, f32, BgF32>(
            queue,
            variant,
            sh,
            plan,
            fl,
            bf16::from_f64,
            bf16::to_f64,
            f64::from,
        ),
    }
}
