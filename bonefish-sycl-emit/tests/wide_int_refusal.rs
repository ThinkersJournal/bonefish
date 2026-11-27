// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unpopped 0.61.3+ `check_kernel` COMPUTES I128/U128/I256/U256; this emitter does not lower them
//! (the carry-flag emit-lowering hold stands), so it must refuse BY NAME before emitting, on both
//! the plan path (`Backend::lower`) and the kernel path (`Backend::lower_kernel`). Toolchain-free.

use bonefish_sycl_emit::Sycl;
use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{OpDef, input};
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, check_kernel};
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const WIDE: [ElementKind; 4] = [
    ElementKind::I128,
    ElementKind::U128,
    ElementKind::I256,
    ElementKind::U256,
];

fn key_for(dt: ElementKind, n: i64) -> unpopped_vocab::StructureKey {
    let d = OperandDesc::new(1, &[n], &[1], dt, 256);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89)
}

fn wide_kernel(dt: ElementKind) -> KernelFn {
    KernelFn::new(
        "wide_store",
        dt,
        vec![Param::buffer("o", dt, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![]),
    )
}

#[test]
fn plan_path_refuses_wide_ints_by_name() {
    let f32op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    assert!(
        try_generate(&f32op, &key_for(ElementKind::F32, 7), &Sycl).is_ok(),
        "positive control: f32 n = 7 lowers"
    );
    for dt in WIDE {
        let op = OpDef::elementwise("add", 2, &[dt], input(0) + input(1));
        match try_generate(&op, &key_for(dt, 7), &Sycl) {
            // Unpopped refuses wide ints at plan time (InadmissiblePlan) before the backend is
            // reached; a backend-level refusal is a typed `UnsupportedDtype`. Either names the dtype.
            Err(LowerError::UnsupportedDtype { dtype, .. }) => assert_eq!(dtype, dt),
            Err(e @ LowerError::InadmissiblePlan { .. }) => {
                assert!(format!("{e:?}").contains(&format!("{dt:?}")), "{e:?}");
            }
            other => panic!("{dt:?}: expected a by-name refusal, got {other:?}"),
        }
    }
}

#[test]
fn kernel_path_refuses_wide_ints_naming_the_dtype() {
    for dt in WIDE {
        let k = wide_kernel(dt);
        eprintln!("{dt:?}: check_kernel = {:?}", check_kernel(&k));
        match Sycl.lower_kernel(&k) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(detail.contains(&format!("{dt:?}")), "{dt:?}: {detail}");
            }
            other => panic!("{dt:?}: expected UnsupportedKernel, got {other:?}"),
        }
    }
}
