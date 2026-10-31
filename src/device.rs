//! Device enumeration and selection (R5a, `docs/FUEL-INTEL-GPU-DESIGN.md`).
//!
//! [`Queue::devices`] lists every SYCL device with what it reports about itself, and a queue can
//! be made on a chosen one by index ([`Queue::with_device`]), by PCI address
//! ([`Queue::with_pci_address`]) or by UUID ([`Queue::with_uuid`]). The point is to pick a
//! *specific* GPU, never "the first one".
//!
//! Information a device does not offer is `None`, never an empty string or a guess: matching on a
//! made-up PCI address or UUID would select the wrong GPU. Whether a given GPU and driver fill the
//! PCI address, UUID and device id in is **UNVERIFIED** (no GPU was available); the CPU device
//! offers none of the Intel-specific ones. The formats are whatever the runtime returns: the PCI
//! address is matched case-insensitively as a string, and the UUID is 32 lowercase hex digits
//! (dashes and case are ignored when matching).
//!
//! # What the one device measured so far offers
//!
//! The only devices available were an OpenCL CPU device (the AMD laptop, Windows, oneAPI 2026.1;
//! the Linux CI runner has the same kind). Findings that should stop anyone treating these fields
//! as hardware identifiers:
//!
//! - the OpenCL name is **padded with trailing spaces** (`"AMD Ryzen 9 7940HX with Radeon
//!   Graphics        "`); [`DeviceInfo::name`] is kept exactly as reported, so compare trimmed;
//! - it reports **no PCI address** (`None`);
//! - it reports a **UUID** and an **"Intel device id"** (10882834 on the laptop) even though it is
//!   an AMD CPU, so the Intel device id is an opaque runtime value on that device, not a PCI
//!   device id;
//! - there was **one** device on both machines, so selecting an index above 0 has never been
//!   exercised here: **selection among several devices is UNVERIFIED**, as is everything a GPU
//!   reports.

use std::ffi::c_char;

use bonefish_sys as sys;

use crate::error::{Error, Result, last_message};
use crate::queue::Queue;

/// What kind of device the runtime says it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceType {
    /// A CPU.
    Cpu,
    /// A GPU.
    Gpu,
    /// An accelerator (FPGA and the like).
    Accelerator,
    /// Anything else the runtime reports, as it spelled it.
    Other(String),
}

/// One SYCL device and what it reports about itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Position in the runtime's device list. Valid only within the process that listed it.
    pub index: usize,
    /// The device name.
    pub name: String,
    /// The vendor string.
    pub vendor: String,
    /// The SYCL backend that exposes it (for example `opencl` or `ext_oneapi_level_zero`).
    pub backend: String,
    /// CPU, GPU, ...
    pub device_type: DeviceType,
    /// The driver version string.
    pub driver_version: String,
    /// PCI address, if the device offers one.
    pub pci_address: Option<String>,
    /// UUID as 32 lowercase hex digits, if the device offers one.
    pub uuid: Option<String>,
    /// Intel device id, if the device offers one.
    pub intel_device_id: Option<u32>,
}

// Field codes of `bf_device_info`.
const NAME: i32 = 0;
const VENDOR: i32 = 1;
const BACKEND: i32 = 2;
const TYPE: i32 = 3;
const DRIVER: i32 = 4;
const PCI: i32 = 5;
const UUID: i32 = 6;
const INTEL_ID: i32 = 7;

fn failure() -> Error {
    if sys::SYCL_AVAILABLE {
        Error::Sycl(last_message())
    } else {
        Error::Unavailable(last_message())
    }
}

/// One field of one device. `Ok(None)` means the device does not offer it.
fn field(index: usize, code: i32) -> Result<Option<String>> {
    let index = i64::try_from(index)
        .map_err(|_| Error::InvalidArgument(format!("device index {index} exceeds i64")))?;
    let mut buf = vec![0 as c_char; 256];
    loop {
        // SAFETY: `buf` is valid for `buf.len()` bytes and the shim NUL-terminates within it.
        let full = unsafe { sys::bf_device_info(index, code, buf.as_mut_ptr(), buf.len() as i64) };
        match full {
            -2 => return Ok(None),
            n if n < 0 => return Err(failure()),
            n if (n as usize) < buf.len() => {
                let bytes: Vec<u8> = buf[..n as usize].iter().map(|c| *c as u8).collect();
                return Ok(Some(String::from_utf8_lossy(&bytes).into_owned()));
            }
            n => buf.resize(n as usize + 1, 0),
        }
    }
}

fn required(index: usize, code: i32) -> Result<String> {
    field(index, code)?.ok_or_else(|| {
        Error::Other(format!(
            "the runtime reported device {index} field {code} as unavailable, which every device must offer"
        ))
    })
}

fn read(index: usize) -> Result<DeviceInfo> {
    let device_type = match required(index, TYPE)?.as_str() {
        "cpu" => DeviceType::Cpu,
        "gpu" => DeviceType::Gpu,
        "accelerator" => DeviceType::Accelerator,
        other => DeviceType::Other(other.to_owned()),
    };
    let intel_device_id = match field(index, INTEL_ID)? {
        Some(s) => Some(s.parse::<u32>().map_err(|_| {
            Error::Other(format!("device {index}: device id {s:?} is not a number"))
        })?),
        None => None,
    };
    Ok(DeviceInfo {
        index,
        name: required(index, NAME)?,
        vendor: required(index, VENDOR)?,
        backend: required(index, BACKEND)?,
        device_type,
        driver_version: required(index, DRIVER)?,
        pci_address: field(index, PCI)?,
        uuid: field(index, UUID)?,
        intel_device_id,
    })
}

/// A UUID as it is compared: lowercase, no dashes.
fn normalise_uuid(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

/// The one device whose PCI address equals `address` (compared case-insensitively).
///
/// # Errors
///
/// [`Error::InvalidArgument`] if no device has that address, or if several do (naming them): a
/// selection that is not unique is refused, not resolved by taking the first.
pub fn find_by_pci_address<'a>(devices: &'a [DeviceInfo], address: &str) -> Result<&'a DeviceInfo> {
    unique(
        devices
            .iter()
            .filter(|d| {
                d.pci_address
                    .as_deref()
                    .is_some_and(|a| a.eq_ignore_ascii_case(address))
            })
            .collect(),
        &format!("PCI address {address:?}"),
        devices,
        |d| d.pci_address.is_some(),
    )
}

/// The one device whose UUID equals `uuid` (case and dashes ignored).
///
/// # Errors
///
/// As [`find_by_pci_address`].
pub fn find_by_uuid<'a>(devices: &'a [DeviceInfo], uuid: &str) -> Result<&'a DeviceInfo> {
    let want = normalise_uuid(uuid);
    unique(
        devices
            .iter()
            .filter(|d| d.uuid.as_deref().is_some_and(|u| normalise_uuid(u) == want))
            .collect(),
        &format!("UUID {uuid:?}"),
        devices,
        |d| d.uuid.is_some(),
    )
}

fn unique<'a>(
    found: Vec<&'a DeviceInfo>,
    what: &str,
    all: &[DeviceInfo],
    offers: impl Fn(&DeviceInfo) -> bool,
) -> Result<&'a DeviceInfo> {
    match found.as_slice() {
        [one] => Ok(one),
        [] => Err(Error::InvalidArgument(format!(
            "no device has {what}; {} of {} devices report one at all",
            all.iter().filter(|d| offers(d)).count(),
            all.len()
        ))),
        many => Err(Error::InvalidArgument(format!(
            "{} devices have {what} (indices {:?}); refusing to pick one",
            many.len(),
            many.iter().map(|d| d.index).collect::<Vec<_>>()
        ))),
    }
}

impl Queue {
    /// Every SYCL device the runtime exposes, in the runtime's order.
    ///
    /// # Errors
    ///
    /// [`Error::Unavailable`] when built without oneAPI; [`Error::Sycl`] if the runtime cannot
    /// enumerate.
    pub fn devices() -> Result<Vec<DeviceInfo>> {
        // SAFETY: plain call.
        let n = unsafe { sys::bf_device_count() };
        if n < 0 {
            return Err(failure());
        }
        (0..n as usize).map(read).collect()
    }

    /// A queue on the device `info` describes.
    ///
    /// The device list is read again first and the device at `info.index` must still be exactly
    /// what `info` says (name, backend, PCI address, UUID, ...). If the list changed since
    /// `info` was made, this is an error rather than a queue on whatever moved into that slot.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an out-of-range index or a stale `info`; [`Error::Sycl`] if
    /// the queue cannot be made.
    pub fn with_device(info: &DeviceInfo) -> Result<Queue> {
        let now = read(info.index)?;
        if &now != info {
            return Err(Error::InvalidArgument(format!(
                "device {} is now {:?}, not {:?}: the device list changed since it was listed",
                info.index, now.name, info.name
            )));
        }
        let index = i64::try_from(info.index)
            .map_err(|_| Error::InvalidArgument("device index exceeds i64".into()))?;
        // SAFETY: plain call; a null return is handled by `from_raw`.
        Queue::from_raw(unsafe { sys::bf_queue_new_device(index) }, false)
    }

    /// As [`Queue::with_device`], with profiling enabled ([`Queue::new_profiling`]).
    ///
    /// # Errors
    ///
    /// As [`Queue::with_device`].
    pub fn with_device_profiling(info: &DeviceInfo) -> Result<Queue> {
        let now = read(info.index)?;
        if &now != info {
            return Err(Error::InvalidArgument(format!(
                "device {} is now {:?}, not {:?}: the device list changed since it was listed",
                info.index, now.name, info.name
            )));
        }
        let index = i64::try_from(info.index)
            .map_err(|_| Error::InvalidArgument("device index exceeds i64".into()))?;
        // SAFETY: plain call; a null return is handled by `from_raw`.
        Queue::from_raw(unsafe { sys::bf_queue_new_ex(0, index, 1) }, true)
    }

    /// A queue on the device at `index` of [`Queue::devices`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if there is no such device.
    pub fn with_device_index(index: usize) -> Result<Queue> {
        let devices = Queue::devices()?;
        let info = devices.get(index).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "device index {index} out of range ({} devices)",
                devices.len()
            ))
        })?;
        Queue::with_device(info)
    }

    /// A queue on the one device with this PCI address (see [`find_by_pci_address`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if no device, or more than one, has that address.
    pub fn with_pci_address(address: &str) -> Result<Queue> {
        let devices = Queue::devices()?;
        Queue::with_device(find_by_pci_address(&devices, address)?)
    }

    /// A queue on the one device with this UUID (see [`find_by_uuid`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if no device, or more than one, has that UUID.
    pub fn with_uuid(uuid: &str) -> Result<Queue> {
        let devices = Queue::devices()?;
        Queue::with_device(find_by_uuid(&devices, uuid)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(index: usize, pci: Option<&str>, uuid: Option<&str>) -> DeviceInfo {
        DeviceInfo {
            index,
            name: format!("device {index}"),
            vendor: "v".into(),
            backend: "b".into(),
            device_type: DeviceType::Gpu,
            driver_version: "1".into(),
            pci_address: pci.map(str::to_owned),
            uuid: uuid.map(str::to_owned),
            intel_device_id: None,
        }
    }

    #[test]
    fn a_pci_address_selects_exactly_one_device_ignoring_case() {
        let devs = [
            dev(0, Some("0000:03:00.0"), None),
            dev(1, Some("0000:0A:00.0"), None),
            dev(2, None, None),
        ];
        // Positive control: the right device is found, in either case.
        assert_eq!(find_by_pci_address(&devs, "0000:0a:00.0").unwrap().index, 1);
        assert_eq!(find_by_pci_address(&devs, "0000:0A:00.0").unwrap().index, 1);
        // Absent: an error that says how many devices report an address at all.
        let e = find_by_pci_address(&devs, "0000:ff:00.0").unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("2 of 3")),
            "{e}"
        );
    }

    #[test]
    fn a_device_without_an_address_is_never_selected_by_one() {
        // A device that offers no PCI address must not match any string, including an empty one.
        let devs = [dev(0, None, None)];
        assert!(find_by_pci_address(&devs, "").is_err());
        assert!(find_by_pci_address(&devs, "0000:00:00.0").is_err());
    }

    #[test]
    fn two_devices_with_the_same_address_are_refused_not_resolved() {
        let devs = [
            dev(0, Some("0000:03:00.0"), None),
            dev(1, Some("0000:03:00.0"), None),
        ];
        let e = find_by_pci_address(&devs, "0000:03:00.0").unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("2 devices") && m.contains("[0, 1]")),
            "{e}"
        );
    }

    #[test]
    fn a_uuid_matches_ignoring_dashes_and_case() {
        let devs = [
            dev(0, None, Some("00112233445566778899aabbccddeeff")),
            dev(1, None, Some("ffeeddccbbaa99887766554433221100")),
        ];
        assert_eq!(
            find_by_uuid(&devs, "00112233-4455-6677-8899-AABBCCDDEEFF")
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            find_by_uuid(&devs, "FFEEDDCC-BBAA-9988-7766-554433221100")
                .unwrap()
                .index,
            1
        );
        assert!(find_by_uuid(&devs, "00000000000000000000000000000000").is_err());
    }
}
