// SPDX-License-Identifier: MIT OR Apache-2.0
//! The SYCL backend lowers a `KernelPlan` (`Backend::lower`) and has no kernel-layer path: it does not
//! override `Backend::lower_kernel`, so a `KernelFn` is refused with the typed
//! `LowerError::UnsupportedKernel` and nothing is generated. That includes unpopped 0.40's
//! `Stmt::AtomicRmw`, which a later kernel-layer path would spell with `sycl::atomic_ref`
//! (`fetch_add` / `fetch_max` / `fetch_min`, relaxed, global address space).
//! Toolchain-free: no SYCL compiler is needed.

use bonefish_sycl_emit::Sycl;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, CollectiveKind, KernelFn, Param, Place, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind;

fn atomic_kernel(op: AtomicOp) -> KernelFn {
    let i32k = ElementKind::I32;
    let mut k = KernelFn::new(
        "atomic_count",
        i32k,
        vec![Param::buffer("c", i32k, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![Stmt::atomic_rmw(
            op,
            Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, i32k)],
            },
            ScalarExpr::ConstInt(1, i32k),
        )]),
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

/// unpopped 0.56.2's NVIDIA-shaped statements are declined BY NAME (CireSnave 2026-11-16 patent note).
#[test]
fn nvidia_sync_and_bulk_copy_statements_are_declined_by_name() {
    let i32k = ElementKind::I32;
    let bar = || Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, i32k)],
    };
    let c = |v: i64| ScalarExpr::ConstInt(v, i32k);
    let cases: Vec<(&str, Stmt)> = vec![
        (
            "MbarInit",
            Stmt::MbarInit {
                bar: bar(),
                count: c(1),
            },
        ),
        (
            "MbarArrive",
            Stmt::MbarArrive {
                bar: bar(),
                tx: None,
            },
        ),
        (
            "MbarCompleteTx",
            Stmt::MbarCompleteTx {
                bar: bar(),
                bytes: c(16),
            },
        ),
        (
            "MbarWait",
            Stmt::MbarWait {
                bar: bar(),
                parity: c(0),
            },
        ),
        (
            "CopyBulk",
            Stmt::CopyBulk {
                dst: bar(),
                src: bar(),
                elems: 4,
                bar: bar(),
            },
        ),
        ("Tcgen05Commit", Stmt::Tcgen05Commit { bar: bar() }),
        ("WgmmaWait", Stmt::WgmmaWait { pending: 0 }),
        ("AsyncWait", Stmt::AsyncWait { pending: 0 }),
        ("AsyncCommit", Stmt::AsyncCommit),
        (
            "CopyAsync",
            Stmt::CopyAsync {
                dst: bar(),
                src: bar(),
                elems: 4,
                valid: None,
                cache: Default::default(),
                l2_prefetch: None,
            },
        ),
        (
            "Collective",
            Stmt::Collective {
                kind: CollectiveKind::Reduce(ReduceOp::Sum),
                value: c(1),
                result: 0,
                scope: Scope::Subgroup,
            },
        ),
        (
            "CollectiveTuple",
            Stmt::CollectiveTuple {
                kind: CollectiveKind::Reduce(ReduceOp::Sum),
                values: vec![c(1)],
                results: vec![0],
                scope: Scope::Subgroup,
            },
        ),
    ];
    let nest = |depth: u8, st: Stmt| -> Stmt {
        // 0: bare; 1: If.then; 2: For inside If.els; 3: While inside For inside If.then.
        let wrap_while = |s: Stmt| Stmt::While {
            cond: c(1),
            body: Block(vec![s]),
        };
        let wrap_for = |s: Stmt| Stmt::For {
            local: 0,
            start: c(0),
            end: c(1),
            step: c(1),
            unroll: Default::default(),
            body: Block(vec![s]),
        };
        match depth {
            0 => st,
            1 => Stmt::If {
                cond: c(1),
                then: Block(vec![st]),
                els: Block(vec![]),
            },
            2 => Stmt::If {
                cond: c(1),
                then: Block(vec![]),
                els: Block(vec![wrap_for(st)]),
            },
            _ => Stmt::If {
                cond: c(1),
                then: Block(vec![wrap_for(wrap_while(st))]),
                els: Block(vec![]),
            },
        }
    };
    for (name, st) in cases {
        for depth in 0..=3u8 {
            let k = KernelFn::new(
                "nv_stmt",
                i32k,
                vec![Param::buffer("c", i32k, BufferRole::ReadWrite, Some(1))],
                vec![],
                Block(vec![nest(depth, st.clone())]),
            );
            match Sycl.lower_kernel(&k) {
                Err(LowerError::UnsupportedKernel { detail }) => {
                    assert!(
                        detail.contains(&format!("Stmt::{name}")),
                        "{name} depth {depth}: {detail}"
                    );
                    assert!(detail.contains("nv_stmt"), "{name} depth {depth}: {detail}");
                }
                other => panic!("{name} depth {depth}: expected UnsupportedKernel, got {other:?}"),
            }
        }
    }
}
