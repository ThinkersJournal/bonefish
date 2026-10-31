// Shared by the shim translation units: export macro, queue handle cast, error capture.
#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <atomic>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define BF_API extern "C" __declspec(dllexport)
#else
#define BF_API extern "C" __attribute__((visibility("default")))
#endif

namespace bf {
// One slot per thread; defined once in shim.cpp, read through bf_last_error().
extern thread_local std::string g_last_error;


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

// Asynchronous SYCL errors (raised after submission, for example by a kernel or a host task) reach
// the queue's async_handler only when the host calls wait_and_throw/throw_asynchronous. The
// handler is the queue's, but it is NOT guaranteed to run on the thread that is waiting: measured
// on Windows (oneAPI 2026.1, CPU device), with a per-thread list four threads on four queues lost
// about half of the errors (138 of 300 on failing queues) and a clean queue sometimes reported
// another queue's error. So the errors are kept per queue, in a store the handler captures, and
// the wait functions drain the store of the queue (or event) they were called on.
struct ErrStore {
  std::mutex m;
  std::vector<std::string> errs;
};

// The handle behind every `void *q` the C API hands out.
struct BfQueue {
  std::shared_ptr<ErrStore> store;
  sycl::queue q;
  template <class Selector>
  BfQueue(const Selector &sel, std::shared_ptr<ErrStore> s, const sycl::property_list &props)
      : store(s), q(sel, make_handler(s), props) {}

  static sycl::async_handler make_handler(std::shared_ptr<ErrStore> s) {
    return [s](sycl::exception_list errs) {
      std::lock_guard<std::mutex> lock(s->m);
      for (const std::exception_ptr &p : errs) {
        try {
          std::rethrow_exception(p);
        } catch (const std::exception &e) {
          s->errs.emplace_back(e.what());
        } catch (...) {
          s->errs.emplace_back("unknown asynchronous exception");
        }
      }
    };
  }
};

// The handle behind every `void *e`: the event, plus the store of the queue it was submitted to,
// so waiting on an event can report that queue's asynchronous errors.
struct BfEvent {
  sycl::event e;
  std::shared_ptr<ErrStore> store;
};

inline sycl::queue &Q(void *q) { return static_cast<BfQueue *>(q)->q; }
inline std::shared_ptr<ErrStore> &Store(void *q) { return static_cast<BfQueue *>(q)->store; }
inline sycl::event &E(void *e) { return static_cast<BfEvent *>(e)->e; }

// 0 when nothing was collected; otherwise 1 (SYCL error) with every message joined in
// g_last_error. Empties the store, so each error is reported once.
inline int32_t drain_async_errors(ErrStore &st) {
  std::lock_guard<std::mutex> lock(st.m);
  if (st.errs.empty()) return 0;
  std::string all;
  for (const std::string &m : st.errs) {
    if (!all.empty()) all += "; ";
    all += m;
  }
  st.errs.clear();
  g_last_error = "asynchronous error: " + all;
  return 1;
}

// Build the dependency vector oneMKL takes from the C array of event handles.
inline std::vector<sycl::event> deps_of(void *const *deps, int64_t n) {
  std::vector<sycl::event> v;
  for (int64_t i = 0; i < n; i++) v.push_back(E(deps[i]));
  return v;
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
