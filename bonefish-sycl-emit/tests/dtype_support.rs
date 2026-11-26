// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Sycl::supports_dtype` computes exactly `f32` and `f64`; every other `ElementKind`, including the
//! seven added by unpopped 0.60.0 (E1a), is declined by name. Toolchain-free.

use bonefish_sycl_emit::Sycl;
use unpopped::backend::{Backend, DtypeSupport};
use unpopped_vocab::{ElementKind, TargetId};

fn target() -> TargetId {
    TargetId::parse("sycl:generic").expect("valid target token")
}

#[test]
fn only_f32_and_f64_are_supported() {
    let supported: Vec<ElementKind> = ElementKind::ALL
        .iter()
        .copied()
        .filter(|k| Sycl.supports_dtype(*k, target()) == DtypeSupport::Supported)
        .collect();
    assert_eq!(supported, vec![ElementKind::F32, ElementKind::F64]);
}

#[test]
fn the_e1a_dtypes_are_declined_by_name() {
    for k in [
        ElementKind::F128,
        ElementKind::I128,
        ElementKind::U128,
        ElementKind::I256,
        ElementKind::U256,
        ElementKind::F8E3M4,
        ElementKind::F8UE4M3,
    ] {
        assert!(ElementKind::ALL.contains(&k), "{k:?} missing from ALL");
        assert_eq!(
            Sycl.supports_dtype(k, target()),
            DtypeSupport::Unsupported,
            "{k:?}"
        );
    }
}
