// SPDX-License-Identifier: MIT OR Apache-2.0
//! The SYCL backend lowers a `KernelPlan` (`Backend::lower`) and has no kernel-layer path: it does not
//! override `Backend::lower_kernel`, so a `KernelFn` is refused with the typed
//! `LowerError::UnsupportedKernel` and nothing is generated. That includes unpopped 0.40's
//! `Stmt::AtomicRmw`, which a later kernel-layer path would spell with `sycl::atomic_ref`
//! (`fetch_add` / `fetch_max` / `fetch_min`, relaxed, global address space).
//! Toolchain-free: no SYCL compiler is needed.

use bonefish_sycl_emit::Sycl;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::LaunchShape;
use unpopped::fixtures::{Fixture, fixtures};
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

/// Statements this backend declines BY NAME (CireSnave 2026-11-16 patent note); the list is spelled out here on
/// purpose so that removing a name from the emitter's own list fails this test (a mutant check).
const DECLINED: &[&str] = &[
    "MbarInit",
    "MbarArrive",
    "MbarCompleteTx",
    "MbarWait",
    "CopyBulk",
    "Tcgen05Commit",
    "WgmmaWait",
    "WgmmaCommit",
    "Tcgen05Ld",
    "ProxyFence",
    "Wgmma",
    "Tcgen05Mma",
    "Tcgen05Alloc",
    "Tcgen05Dealloc",
    "Tcgen05RelinquishPermit",
    "CopyUnpack",
    "AsyncWait",
    "CopyAsync",
    "AsyncCommit",
    "Collective",
    "CollectiveTuple",
];

/// The kernel a fixture lowers: the one kernel, or the entry function of the module.
fn entry(f: &Fixture) -> KernelFn {
    match f {
        Fixture::Kernel(k) => k.clone(),
        Fixture::Module { module, entry } => module.fns[*entry as usize].clone(),
        other => panic!("unknown fixture shape {other:?}"),
    }
}

fn reason(k: &KernelFn) -> String {
    match Sycl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("{}: expected UnsupportedKernel, got {other:?}", k.name),
    }
}

/// Every `Stmt` variant has a checker-valid fixture in unpopped (feature `fixtures`); each is refused, and the
/// ones in [`DECLINED`] are refused BY NAME, bare and nested under `If` / `For` / `While`.
#[test]
fn every_stmt_fixture_is_valid_and_refused_and_declined_ones_by_name() {
    let i32k = ElementKind::I32;
    let c = |v: i64| ScalarExpr::ConstInt(v, i32k);
    let all = fixtures();
    assert!(all.len() >= DECLINED.len(), "{} fixtures", all.len());
    for d in DECLINED {
        assert!(all.iter().any(|(n, _)| n == d), "no fixture named {d}");
    }
    for (name, fx) in &all {
        assert_eq!(
            fx.check(),
            Ok(()),
            "{name}: the fixture must be checker-valid"
        );
        let k = entry(fx);
        let wrap = |depth: u8| -> KernelFn {
            let mut w = k.clone();
            let inner = Block(k.body.0.clone());
            w.body = Block(vec![match depth {
                0 => return k.clone(),
                1 => Stmt::If {
                    cond: c(1),
                    then: inner,
                    els: Block(vec![]),
                },
                2 => Stmt::If {
                    cond: c(1),
                    then: Block(vec![]),
                    els: Block(vec![Stmt::While {
                        cond: c(1),
                        body: inner,
                    }]),
                },
                _ => Stmt::If {
                    cond: c(1),
                    then: Block(vec![Stmt::While {
                        cond: c(1),
                        body: Block(vec![Stmt::While {
                            cond: c(1),
                            body: inner,
                        }]),
                    }]),
                    els: Block(vec![]),
                },
            }]);
            w
        };
        for depth in 0..=3u8 {
            let d = reason(&wrap(depth));
            assert!(d.contains("sycl"), "{name} depth {depth}: {d}");
            assert!(d.contains(&k.name), "{name} depth {depth}: {d}");
            if DECLINED.contains(name) {
                assert!(
                    d.contains(&format!("declines `Stmt::{name}` by name")),
                    "{name} depth {depth}: {d}"
                );
            } else {
                assert!(
                    !d.contains("declines `Stmt::"),
                    "{name} depth {depth}: must not be named: {d}"
                );
            }
        }
    }
}

/// unpopped 0.58.4 added the exclusive Max / Min / Prod scans to `Stmt::Collective`: a kernel using one is
/// checker-valid and is declined by the name `Collective`, bare and nested.
#[test]
fn exclusive_max_min_prod_scans_are_declined_by_name_collective() {
    let (_, fx) = fixtures()
        .into_iter()
        .find(|(n, _)| *n == "Collective")
        .expect("Collective fixture");
    let base = entry(&fx);
    for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod] {
        for exclusive in [true, false] {
            let mut k = base.clone();
            let mut hit = 0;
            for st in &mut k.body.0 {
                if let Stmt::Collective { kind, .. } = st {
                    *kind = CollectiveKind::Scan { op, exclusive };
                    hit += 1;
                }
            }
            assert_eq!(hit, 1, "{op:?}: anchor");
            assert_eq!(
                check_kernel(&k),
                Ok(()),
                "{op:?} exclusive={exclusive}: must be checker-valid"
            );
            let d = reason(&k);
            assert!(
                d.contains("declines `Stmt::Collective` by name"),
                "{op:?} exclusive={exclusive}: {d}"
            );
        }
    }
}
