// Shared by the shim translation units: export macro, queue handle cast, error capture.
#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>

#if defined(_WIN32)
#define BF_API extern "C" __declspec(dllexport)
#else
#define BF_API extern "C" __attribute__((visibility("default")))
#endif

namespace bf {
// One slot per thread; defined once in shim.cpp, read through bf_last_error().
extern thread_local std::string g_last_error;

inline sycl::queue &Q(void *q) { return *static_cast<sycl::queue *>(q); }

// Run f, mapping exceptions to status codes: 0 ok, 1 SYCL error, 2 oneMKL error, 3 other.
template <class F> int32_t guard(F &&f) {
  try {
    f();
    return 0;
  } catch (const sycl::exception &e) {
    g_last_error = e.what();
    return 1;
  } catch (const oneapi::mkl::exception &e) {
    g_last_error = e.what();
    return 2;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return 3;
  } catch (...) {
    g_last_error = "unknown C++ exception";
    return 3;
  }
}

// One process-wide lock around every oneMKL entry point.
//
// Measured, in a standalone C++ program with no bonefish code (oneAPI 2026.1, Windows, CPU/OpenCL
// device): four threads each calling `oneapi::mkl::blas::column_major::gemm`, 40 runs. Cold, the
// first concurrent use of oneMKL crashes inside `mkl_sycl_blas.6.dll` in 28 of 40 runs (exception
// 0xC06D007F or an access violation). The same program passes 40 of 40 when the calls take one
// mutex, 40 of 40 when one queue is warmed by a serial call first (so the race is in process-wide
// first-use initialisation, not per-queue state), and SYCL allocation under threads alone passes
// 25 of 25. The oneAPI specification says oneMKL functions "shall be host thread safe"; this
// shipped build did not honour that. Serialising the calls makes thread-safety ours by
// construction instead of Intel's claim. SYCL queue creation, allocation and copies are not locked.
// Revisit when a newer oneMKL passes the unlocked stress test.
inline std::mutex &mkl_mutex() {
  static std::mutex m;
  return m;
}

// Diagnostic switch, not a feature: BONEFISH_UNSAFE_NO_MKL_LOCK=1 turns the lock off so the same
// binary can show whether a given oneMKL still needs it (the baseline in the stress comparison, and
// the retest for a newer oneAPI). Read once.
inline bool mkl_lock_enabled() {
  static const bool enabled = [] {
    const char *v = std::getenv("BONEFISH_UNSAFE_NO_MKL_LOCK");
    return !(v && v[0] == '1' && v[1] == '\0');
  }();
  return enabled;
}

// How many times a oneMKL call has entered the lock path. Exported (`bf_mkl_lock_acquisitions`)
// only so a test can prove that every oneMKL-calling shim function goes through `MklLock`: a
// function that bypasses it leaves this counter unchanged. It counts entries, so it still moves
// when the diagnostic switch disables the lock itself.
inline std::atomic<int64_t> &mkl_lock_acquisitions() {
  static std::atomic<int64_t> n{0};
  return n;
}

// Holds the oneMKL lock for its lifetime (unless the diagnostic switch is set).
class MklLock {
public:
  MklLock() : lock_(mkl_mutex(), std::defer_lock) {
    mkl_lock_acquisitions().fetch_add(1, std::memory_order_relaxed);
    if (mkl_lock_enabled()) lock_.lock();
  }

private:
  std::unique_lock<std::mutex> lock_;
};

// `guard`, holding the oneMKL lock for the duration of the call.
template <class F> int32_t guard_mkl(F &&f) {
  MklLock lock;
  return guard(std::forward<F>(f));
}
} // namespace bf
