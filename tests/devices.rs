//! Device enumeration and selection (R5a). Needs a device; without oneAPI the listing must fail
//! with `Unavailable` (and with `BONEFISH_REQUIRE_SYCL=1` a build without SYCL is a failure). The
//! CPU device is what CI and the laptop have; **no GPU has run any of this** (UNVERIFIED), so what
//! a GPU reports for its PCI address, UUID and device id is not tested here.

use bonefish::{
    DeviceInfo, DeviceType, Error, Queue, SYCL_AVAILABLE, UsmBuffer, UsmKind, blas,
    find_by_pci_address,
};

/// The listed devices, or `None` after a loud skip / the `Unavailable` check.
fn listed(test: &str) -> Option<Vec<DeviceInfo>> {
    if !SYCL_AVAILABLE {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but bonefish was built without oneAPI"
        );
        assert!(
            matches!(Queue::devices(), Err(Error::Unavailable(_))),
            "{test}: without oneAPI the listing must say so"
        );
        eprintln!("SKIP {test}: built without oneAPI. Nothing ran on a device.");
        return None;
    }
    Some(Queue::devices().expect("device enumeration"))
}

fn cpu(devices: &[DeviceInfo]) -> &DeviceInfo {
    devices
        .iter()
        .find(|d| d.device_type == DeviceType::Cpu)
        .expect("a CPU SYCL device (the laptop and the CI runner have one)")
}

#[test]
fn devices_are_listed_with_their_identifying_fields() {
    let Some(devices) = listed("list") else {
        return;
    };
    assert!(!devices.is_empty());
    for (i, d) in devices.iter().enumerate() {
        assert_eq!(d.index, i, "indices run 0..n in order");
        for (what, v) in [
            ("name", &d.name),
            ("vendor", &d.vendor),
            ("backend", &d.backend),
            ("driver version", &d.driver_version),
        ] {
            assert!(!v.is_empty(), "device {i}: empty {what}");
        }
        // The UUID, when offered, is 32 lowercase hex digits by construction of the shim.
        if let Some(u) = &d.uuid {
            assert!(
                u.len() == 32
                    && u.chars()
                        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "device {i}: UUID {u:?}"
            );
        }
        // An offered PCI address is never empty (an empty one would match nothing but invite
        // matching on nothing).
        if let Some(p) = &d.pci_address {
            assert!(!p.is_empty(), "device {i}: empty PCI address");
        }
    }
    // Report what the CPU device offers, so the CI log shows it (not an assertion about it).
    let c = cpu(&devices);
    eprintln!(
        "CPU device: {:?} backend {} pci {:?} uuid {:?} intel id {:?}",
        c.name, c.backend, c.pci_address, c.uuid, c.intel_device_id
    );
}

#[test]
fn a_queue_on_a_listed_device_is_that_device_and_computes() {
    let Some(devices) = listed("with_device") else {
        return;
    };
    let info = cpu(&devices);
    let q = Queue::with_device(info).expect("a queue on the CPU device");
    assert_eq!(
        q.device_name().unwrap(),
        info.name,
        "the queue must be on the device that was asked for"
    );
    // And it works: x . x = 1 + 4 + 9.
    let x = UsmBuffer::from_slice(&q, UsmKind::Shared, &[1.0_f64, 2.0, 3.0]).unwrap();
    assert_eq!(blas::dot(&q, 3, &x, 1, &x, 1).unwrap(), 14.0);
    // The same through the index form.
    let q2 = Queue::with_device_index(info.index).unwrap();
    assert_eq!(q2.device_name().unwrap(), info.name);
}

#[test]
fn a_device_that_does_not_exist_is_an_error_never_a_default_queue() {
    let Some(devices) = listed("missing") else {
        return;
    };
    // Control: the last real index works.
    assert!(Queue::with_device_index(devices.len() - 1).is_ok());
    let e = Queue::with_device_index(devices.len()).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("out of range")),
        "{e}"
    );
    let e = Queue::with_pci_address("ffff:ff:ff.f").unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
    let e = Queue::with_uuid("00000000000000000000000000000000").unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e}");
}

#[test]
fn a_stale_description_is_refused() {
    let Some(devices) = listed("stale") else {
        return;
    };
    let mut info = cpu(&devices).clone();
    // Control: the unchanged description is accepted.
    assert!(Queue::with_device(&info).is_ok());
    info.name.push_str(" (not me)");
    let e = Queue::with_device(&info).unwrap_err();
    assert!(
        matches!(&e, Error::InvalidArgument(m) if m.contains("changed")),
        "{e}"
    );
}

#[test]
fn matching_by_pci_address_uses_only_what_a_device_offers() {
    let Some(devices) = listed("pci") else {
        return;
    };
    for d in &devices {
        match &d.pci_address {
            // Offered: it finds this device (or refuses if another device claims the same one).
            Some(p) => match find_by_pci_address(&devices, p) {
                Ok(found) => assert_eq!(found.index, d.index),
                Err(Error::InvalidArgument(m)) => assert!(m.contains("devices have"), "{m}"),
                Err(e) => panic!("{e}"),
            },
            // Not offered: an empty address must not select it.
            None => assert!(find_by_pci_address(&devices, "").is_err()),
        }
    }
}
