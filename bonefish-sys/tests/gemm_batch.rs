//! Strided batched `gemm` (non-blocking) through the shim on the CPU (OpenCL) device, single and
//! double precision, against hand-computed 2 x 2 results over a batch of 2. Compiled out without
//! oneAPI; the oneAPI CI job sets `BONEFISH_REQUIRE_SYCL=1` so that cannot hide a skipped run.
#![cfg(bonefish_sycl)]

use bonefish_sys::*;
use std::ffi::{CStr, c_void};

fn err() -> String {
    unsafe { CStr::from_ptr(bf_last_error()) }
        .to_string_lossy()
        .into_owned()
}

macro_rules! batch_test {
    ($name:ident, $t:ty, $f:ident) => {
        #[test]
        fn $name() {
            let q = unsafe { bf_queue_new(1) };
            assert!(!q.is_null(), "no CPU SYCL device: {}", err());
            // Column-major. Batch 0: A0 = [[1,2],[3,4]], B0 = I. Batch 1: A1 = 2 I, B1 = [[1,2],[3,4]].
            // Expected C0 = A0, C1 = 2 * B1 = [[2,4],[6,8]]. Strides are 4 elements, c starts at 0.
            let a: [$t; 8] = [1.0, 3.0, 2.0, 4.0, 2.0, 0.0, 0.0, 2.0];
            let b: [$t; 8] = [1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 2.0, 4.0];
            let bytes = (8 * std::mem::size_of::<$t>()) as i64;
            let alloc = |v: &[$t; 8]| {
                let p = unsafe { bf_malloc(q, 0, bytes) }.cast::<$t>();
                assert!(!p.is_null(), "{}", err());
                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p, 8) };
                p
            };
            let (pa, pb, pc) = (alloc(&a), alloc(&b), alloc(&[0.0; 8]));
            let ev = unsafe {
                $f(
                    q,
                    0,
                    0,
                    2,
                    2,
                    2,
                    1.0,
                    pa,
                    2,
                    4,
                    pb,
                    2,
                    4,
                    0.0,
                    pc,
                    2,
                    4,
                    2,
                    std::ptr::null(),
                    0,
                )
            };
            assert!(!ev.is_null(), "no event: {}", err());
            assert_eq!(unsafe { bf_event_wait(ev) }, 0, "{}", err());
            unsafe { bf_event_free(ev) };
            let got = unsafe { std::slice::from_raw_parts(pc, 8) }.to_vec();
            assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0, 2.0, 6.0, 4.0, 8.0]);
            for p in [pa, pb, pc] {
                unsafe { bf_free(q, p.cast::<c_void>()) };
            }
            unsafe { bf_queue_free(q) };
        }
    };
}

batch_test!(sgemm_batch_strided, f32, bf_sgemm_batch_strided_async);
batch_test!(dgemm_batch_strided, f64, bf_dgemm_batch_strided_async);
