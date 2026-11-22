//! Half-precision `gemm` (non-blocking) through the shim on the CPU (OpenCL) device: the four
//! oneMKL variants against a hand-computed 2 x 2 result. Compiled out without oneAPI; the oneAPI
//! CI job sets `BONEFISH_REQUIRE_SYCL=1`. A device that does not run a variant is REPORTED as
//! "NOT RUN" (stderr, loud) and the test returns, never passes silently as if it had run; set
//! `BONEFISH_REQUIRE_HALF=1` to make that a failure.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};

fn err() -> String {
    unsafe { CStr::from_ptr(bf_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// IEEE binary16 bits of a small exactly-representable value (normal numbers and zero only).
fn f16_bits(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "{x} is not a normal binary16");
    assert_eq!(b & 0x1fff, 0, "{x} is not exactly representable in binary16");
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

/// bfloat16 bits: the upper 16 bits of the f32 (exact for the values used here).
fn bf16_bits(x: f32) -> u16 {
    assert_eq!(
        x.to_bits() & 0xffff,
        0,
        "{x} is not exactly representable in bf16"
    );
    (x.to_bits() >> 16) as u16
}

fn f16_value(h: u16) -> f32 {
    if h & 0x7fff == 0 {
        return 0.0;
    }
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32 - 15;
    let man = (h & 0x3ff) as f32 / 1024.0;
    sign * (1.0 + man) * 2f32.powi(exp)
}

fn bf16_value(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

// Column-major. A = [[1,2],[3,4]], B = [[1,0],[1,1]]; alpha = 2: C = 2 * A * B = [[6,4],[14,8]].
const A: [f32; 4] = [1.0, 3.0, 2.0, 4.0];
const B: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
const WANT: [f32; 4] = [6.0, 14.0, 4.0, 8.0];

type Half16 = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i64,
    i64,
    i64,
    f32,
    *const u16,
    i64,
    *const u16,
    i64,
    f32,
    *mut u16,
    i64,
    *const *mut c_void,
    i64,
) -> *mut c_void;
type HalfF32 = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i64,
    i64,
    i64,
    f32,
    *const u16,
    i64,
    *const u16,
    i64,
    f32,
    *mut f32,
    i64,
    *const *mut c_void,
    i64,
) -> *mut c_void;

fn shared<T: Copy>(q: *mut c_void, v: &[T]) -> *mut T {
    let p = unsafe { bf_malloc(q, 0, std::mem::size_of_val(v) as i64) }.cast::<T>();
    assert!(!p.is_null(), "{}", err());
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, v.len()) };
    p
}

/// `true` if the device ran the variant to completion; `false` (loudly) if it refused.
fn finish(ev: *mut c_void, name: &str) -> bool {
    if ev.is_null() {
        let m = err();
        eprintln!("NOT RUN {name}: this device did not run the variant: {m}");
        assert!(
            !std::env::var("BONEFISH_REQUIRE_HALF").is_ok_and(|v| v == "1"),
            "{name}: BONEFISH_REQUIRE_HALF=1 but the variant failed: {m}"
        );
        return false;
    }
    assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{name}: {}", err());
    unsafe { bf_event_free(ev) };
    true
}

fn run16(name: &str, f: Half16, enc: fn(f32) -> u16, dec: fn(u16) -> f32) {
    let q = unsafe { bf_queue_new(1) };
    assert!(!q.is_null(), "no CPU SYCL device: {}", err());
    let (a, b) = (shared(q, &A.map(enc)), shared(q, &B.map(enc)));
    let c = shared(q, &[0u16; 4]);
    let ev = unsafe { f(q, 0, 0, 2, 2, 2, 2.0, a, 2, b, 2, 0.0, c, 2, std::ptr::null(), 0) };
    if finish(ev, name) {
        let got: Vec<f32> = unsafe { std::slice::from_raw_parts(c, 4) }
            .iter()
            .map(|&h| dec(h))
            .collect();
        assert_eq!(got, WANT.to_vec(), "{name}");
        eprintln!("RAN {name}");
    }
    for p in [a.cast::<c_void>(), b.cast(), c.cast()] {
        unsafe { bf_free(q, p) };
    }
    unsafe { bf_queue_free(q) };
}

fn run_f32(name: &str, f: HalfF32, enc: fn(f32) -> u16) {
    let q = unsafe { bf_queue_new(1) };
    assert!(!q.is_null(), "no CPU SYCL device: {}", err());
    let (a, b) = (shared(q, &A.map(enc)), shared(q, &B.map(enc)));
    let c = shared(q, &[0f32; 4]);
    let ev = unsafe { f(q, 0, 0, 2, 2, 2, 2.0, a, 2, b, 2, 0.0, c, 2, std::ptr::null(), 0) };
    if finish(ev, name) {
        assert_eq!(unsafe { std::slice::from_raw_parts(c, 4) }, &WANT, "{name}");
        eprintln!("RAN {name}");
    }
    for p in [a.cast::<c_void>(), b.cast(), c.cast()] {
        unsafe { bf_free(q, p) };
    }
    unsafe { bf_queue_free(q) };
}

#[test]
fn hgemm_half_output() {
    run16("bf_hgemm_async", bf_hgemm_async, f16_bits, f16_value);
}

#[test]
fn hgemm_f32_output() {
    run_f32("bf_hgemm_f32_async", bf_hgemm_f32_async, f16_bits);
}

#[test]
fn bgemm_bf16_output() {
    run16("bf_bgemm_async", bf_bgemm_async, bf16_bits, bf16_value);
}

#[test]
fn bgemm_f32_output() {
    run_f32("bf_bgemm_f32_async", bf_bgemm_f32_async, bf16_bits);
}
