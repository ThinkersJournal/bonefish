// Shared by the shim translation units: export macro, queue handle cast, error capture.
#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <atomic>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
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
  // The previous asynchronous submission on this queue (see `deps_chained`). Guarded by `lm`.
  std::mutex lm;
  sycl::event last;
  bool has_last = false;
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

// Diagnostic switch, not a feature: BONEFISH_UNSAFE_NO_ASYNC_CHAIN=1 (or `bf_test_set_async_chain`)
// turns the submission chain below off, so the same binary can show whether a given oneMKL still
// needs it. Default on.
inline std::atomic<bool> &async_chain_flag() {
  static std::atomic<bool> on{[] {
    const char *v = std::getenv("BONEFISH_UNSAFE_NO_ASYNC_CHAIN");
    return !(v && v[0] == '1' && v[1] == '\0');
  }()};
  return on;
}

// Every asynchronous submission on a queue depends on the queue's previous asynchronous
// submission, in addition to the caller's `deps`. Measured (Windows, oneAPI 2026.1, CPU/OpenCL,
// in-order queue): oneMKL `dot` submitted right after `axpy`, `copy` or `gemm` read the OLD data in
// 299 of 300, 199 of 200 and 200 of 200 rounds, and 0 of 300 when it listed the earlier event as a
// dependency. So an in-order queue does not, by itself, order oneMKL calls, and the safe layer's
// soundness (no two operations on one buffer overlap on the device) cannot rest on it. Chaining
// the events makes the order ours instead of the library's. Callers hold the oneMKL lock.
inline std::vector<sycl::event> deps_chained(ErrStore &st, void *const *deps, int64_t n) {
  std::vector<sycl::event> v = deps_of(deps, n);
  if (async_chain_flag().load()) {
    std::lock_guard<std::mutex> lock(st.lm);
    if (st.has_last) v.push_back(st.last);
  }
  return v;
}

// Record `e` as the queue's latest asynchronous submission and wrap it as a C handle.
inline void *new_event(sycl::event e, const std::shared_ptr<ErrStore> &st) {
  {
    std::lock_guard<std::mutex> lock(st->lm);
    st->last = e;
    st->has_last = true;
  }
  return new BfEvent{e, st};
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

// Refuse an unusable DFT shape before it reaches oneMKL. A non-positive length must never be handed
// to oneMKL: committing a descriptor of length 0 crashes the process with SIGFPE (exit 136) on Intel
// Xeon CPUs under default instruction-set dispatch (measured: Platinum 8370C and 8573C, oneAPI
// 2026.2, Linux; the same call throws FFT_INVALID_DESCRIPTOR on AMD EPYC, or with
// MKL_ENABLE_INSTRUCTIONS=AVX2). ROADMAP R9, docs/INTEL-REPORT-DRAFT.md finding 3. Used by every
// shim function that builds a DFT descriptor (complex and real).
inline void check_dft_lengths(int64_t ndim, const int64_t *lengths) {
  if (ndim < 1 || lengths == nullptr) throw std::invalid_argument("dft: need at least one length");
  for (int64_t i = 0; i < ndim; i++) {
    if (lengths[i] < 1) {
      throw std::invalid_argument(
          "dft: every length must be at least 1 (a length of 0 or less is refused before oneMKL, "
          "which crashes on it on some Intel CPUs)");
    }
  }
}

// `guard`, holding the oneMKL lock for the duration of the call.
template <class F> int32_t guard_mkl(F &&f) {
  MklLock lock;
  return guard(std::forward<F>(f));
}

// The end of every blocking entry point (ROADMAP R3d): wait for the queue and collect its
// asynchronous errors, so a failure that surfaces during the call is reported by it (status 1)
// instead of staying in the store for some later wait. Waits for the whole queue, not just this
// call's event, because `wait_and_throw` is what delivers the errors; on an in-order queue that is
// the same set of work. Run after the oneMKL lock is released.
inline int32_t settle(sycl::queue &q, ErrStore &st) {
  int32_t rc = guard([&] { q.wait_and_throw(); });
  int32_t arc = drain_async_errors(st);
  return rc != 0 ? rc : arc;
}

// `guard_mkl`, then `settle` when the call itself succeeded.
template <class F> int32_t guard_blocking(sycl::queue &q, ErrStore &st, F &&f) {
  int32_t rc = guard_mkl(std::forward<F>(f));
  return rc != 0 ? rc : settle(q, st);
}
} // namespace bf
