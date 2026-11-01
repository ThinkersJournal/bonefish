//! The SYCL backend lowers a `KernelPlan` (`Backend::lower`) and has no kernel-layer path: it does not
//! override `Backend::lower_kernel`, so a `KernelFn` is refused with the typed
//! `LowerError::UnsupportedKernel` and nothing is generated. That includes unpopped 0.40's
//! `Stmt::AtomicRmw`, which a later kernel-layer path would spell with `sycl::atomic_ref`
//! (`fetch_add` / `fetch_max` / `fetch_min`, relaxed, global address space).
//! Toolchain-free: no SYCL compiler is needed.

use bonefish_sycl_emit::Sycl;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::LaunchShape;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{AtomicOp, Block, BufferRole, KernelFn, Param, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

fn atomic_kernel(op: AtomicOp) -> KernelFn {
    let i32k = ElementKind::I32;
    let mut k = KernelFn::new(
        "atomic_count",
        i32k,
        vec![Param::buffer("c", i32k, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![Stmt::AtomicRmw {
            op,
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, i32k)],
            },
            value: ScalarExpr::ConstInt(1, i32k),
        }]),
    );
    k.launch = Some(LaunchShape::new([4, 1, 1], [3, 1, 1]));
    k
}

#[test]
fn a_kernel_with_an_atomic_is_refused_with_a_named_reason() {
    for op in [AtomicOp::Add, AtomicOp::Max, AtomicOp::Min] {
        let k = atomic_kernel(op);
        // The kernel is valid, so the refusal is the backend's, not the checker's.
        assert_eq!(check_kernel(&k), Ok(()), "{op:?}");
        match Sycl.lower_kernel(&k) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(detail.contains("sycl"), "{detail}");
                assert!(detail.contains("atomic_count"), "{detail}");
            }
            other => panic!("{op:?}: expected UnsupportedKernel, got {other:?}"),
        }
    }
}
