// Device enumeration and selection (R5a, docs/FUEL-INTEL-GPU-DESIGN.md): list every SYCL device,
// read identifying information where the device offers it, and create a queue on a chosen one.
// None of this calls oneMKL, so none of it takes the oneMKL lock.
//
// Information that a device does not offer is reported as "not available" (-2), never as an empty
// string or a guess: the caller matches devices by PCI address or UUID and a made-up value there
// would select the wrong GPU. Whether a given GPU and driver fill these in is UNVERIFIED (no GPU
// was available when this was written; the CPU device offers none of the Intel-specific ones).
#include "common.hpp"

#include <array>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
// The runtime's device list. Enumerated again on every call; the order is stable within a process
// as far as observed, and `bf_queue_new_device` is documented to be used with an index from the
// same process.
std::vector<sycl::device> devices() { return sycl::device::get_devices(); }

std::string hex_uuid(const std::array<unsigned char, 16> &u) {
  static const char digits[] = "0123456789abcdef";
  std::string s;
  for (unsigned char b : u) {
    s += digits[b >> 4];
    s += digits[b & 15];
  }
  return s;
}

const char *type_name(sycl::info::device_type t) {
  switch (t) {
  case sycl::info::device_type::cpu: return "cpu";
  case sycl::info::device_type::gpu: return "gpu";
  case sycl::info::device_type::accelerator: return "accelerator";
  case sycl::info::device_type::custom: return "custom";
  case sycl::info::device_type::automatic: return "automatic";
  case sycl::info::device_type::host: return "host";
  case sycl::info::device_type::all: return "all";
  }
  return "other";
}
} // namespace

// Number of SYCL devices, or -1 on failure (see bf_last_error).
BF_API int64_t bf_device_count() {
  int64_t n = -1;
  bf::guard([&] { n = static_cast<int64_t>(devices().size()); });
  return n;
}

// One piece of information about device `index`, as a NUL-terminated string copied into `buf`
// (truncated to `cap`). Returns the full length, -1 on failure (bad index or runtime error, see
// bf_last_error), or -2 when this device does not offer that information.
// field: 0 name, 1 vendor, 2 backend, 3 device type (cpu/gpu/accelerator/...), 4 driver version,
// 5 PCI address, 6 UUID (32 lowercase hex digits), 7 Intel device id (decimal).
BF_API int64_t bf_device_info(int64_t index, int32_t field, char *buf, int64_t cap) {
  std::string out;
  int64_t status = 0;
  int32_t rc = bf::guard([&] {
    auto all = devices();
    if (index < 0 || static_cast<size_t>(index) >= all.size()) {
      throw std::out_of_range("device index " + std::to_string(index) + " out of range (" +
                              std::to_string(all.size()) + " devices)");
    }
    const sycl::device &d = all[static_cast<size_t>(index)];
    switch (field) {
    case 0: out = d.get_info<sycl::info::device::name>(); break;
    case 1: out = d.get_info<sycl::info::device::vendor>(); break;
    case 2: {
      std::ostringstream os;
      os << d.get_backend();
      out = os.str();
      break;
    }
    case 3: out = type_name(d.get_info<sycl::info::device::device_type>()); break;
    case 4: out = d.get_info<sycl::info::device::driver_version>(); break;
    case 5:
      if (!d.has(sycl::aspect::ext_intel_pci_address)) { status = -2; break; }
      out = d.get_info<sycl::ext::intel::info::device::pci_address>();
      break;
    case 6:
      if (!d.has(sycl::aspect::ext_intel_device_info_uuid)) { status = -2; break; }
      out = hex_uuid(d.get_info<sycl::ext::intel::info::device::uuid>());
      break;
    case 7:
      if (!d.has(sycl::aspect::ext_intel_device_id)) { status = -2; break; }
      out = std::to_string(d.get_info<sycl::ext::intel::info::device::device_id>());
      break;
    default: throw std::invalid_argument("unknown device info field " + std::to_string(field));
    }
  });
  if (rc != 0) return -1;
  if (status != 0) return status;
  if (buf && cap > 0) {
    std::strncpy(buf, out.c_str(), static_cast<size_t>(cap - 1));
    buf[cap - 1] = '\0';
  }
  return static_cast<int64_t>(out.size());
}

// A queue (in-order, with the async handler, like bf_queue_new) on device `index` of the same
// enumeration. Null on failure (bad index or runtime error, see bf_last_error).
BF_API void *bf_queue_new_device(int64_t index) {
  void *out = nullptr;
  bf::guard([&] {
    auto all = devices();
    if (index < 0 || static_cast<size_t>(index) >= all.size()) {
      throw std::out_of_range("device index " + std::to_string(index) + " out of range (" +
                              std::to_string(all.size()) + " devices)");
    }
    const sycl::property_list props{sycl::property::queue::in_order()};
    auto store = std::make_shared<bf::ErrStore>();
    out = new bf::BfQueue{all[static_cast<size_t>(index)], store, props};
  });
  return out;
}

// A queue with options. `device_index` >= 0 picks that device of the enumeration above; -1 picks by
// `kind` (0 default selector, 1 cpu, 2 gpu), like bf_queue_new. `flags` bit 0: enable profiling
// (property::queue::enable_profiling), which `bf_event_profiling` needs. Always in-order and with
// the async handler. Null on failure (see bf_last_error).
BF_API void *bf_queue_new_ex(int32_t kind, int64_t device_index, int32_t flags) {
  void *out = nullptr;
  bf::guard([&] {
    sycl::property_list props{sycl::property::queue::in_order()};
    if (flags & 1) {
      props = sycl::property_list{sycl::property::queue::in_order(),
                                  sycl::property::queue::enable_profiling()};
    }
    auto store = std::make_shared<bf::ErrStore>();
    if (device_index >= 0) {
      auto all = devices();
      if (static_cast<size_t>(device_index) >= all.size()) {
        throw std::out_of_range("device index " + std::to_string(device_index) +
                                " out of range (" + std::to_string(all.size()) + " devices)");
      }
      out = new bf::BfQueue{all[static_cast<size_t>(device_index)], store, props};
      return;
    }
    switch (kind) {
    case 1: out = new bf::BfQueue{sycl::cpu_selector_v, store, props}; break;
    case 2: out = new bf::BfQueue{sycl::gpu_selector_v, store, props}; break;
    default: out = new bf::BfQueue{sycl::default_selector_v, store, props}; break;
    }
  });
  return out;
}
