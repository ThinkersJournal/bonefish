// extern "C" shim over oneapi::mkl (SYCL). Opaque handles only; no C++ type crosses the boundary.
#include "common.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

thread_local std::string bf::g_last_error;

using bf::guard;
using bf::Q;

namespace {
// Freed queues are RETIRED, not destroyed at once, and destroyed only after a grace period (and
// after a wait_and_throw). Measured (Windows, oneAPI 2026.1, CPU device): with four threads each
// looping {new queue, a failing host task, one wait, free the queue}, destroying the queue at once
// killed the process in 25 of 25 runs ("Default async_handler caught exceptions ... terminate
// called"), because a wait can return a little BEFORE the runtime posts the task's error; the
// error then arrives at a queue that no longer exists, goes to SYCL's default handler, and that
// terminates the process. One thread: 0 of 25. With the grace period the 4-thread run survived
// (see ROADMAP R10 for the numbers and what is not shown). The grace period is a heuristic about
// how late an error can be posted, not a proof; the real bound is unknown.
//
// The retired list is leaked on purpose and never destroyed at process exit: destroying SYCL
// objects from a static destructor races the runtime's own shutdown.
using Clock = std::chrono::steady_clock;

struct Retired {
  std::unique_ptr<bf::BfQueue> q;
  Clock::time_point at;
};

std::mutex &retired_mutex() {
  static auto *m = new std::mutex;
  return *m;
}

std::vector<Retired> &retired() {
  static auto *v = new std::vector<Retired>;
  return *v;
}

// BONEFISH_UNSAFE_QUEUE_GRACE_MS overrides the grace period (milliseconds). A diagnostic for
// A/B runs, not a feature: 0 destroys a queue as soon as it has been waited on.
Clock::duration grace() {
  static const Clock::duration g = [] {
    long ms = 2000;
    if (const char *e = std::getenv("BONEFISH_UNSAFE_QUEUE_GRACE_MS")) {
      char *end = nullptr;
      long v = std::strtol(e, &end, 10);
      if (end != e && v >= 0) ms = v;
    }
    return std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(ms));
  }();
  return g;
}

// Destroy every retired queue whose grace period is over: wait for it first, so its work finishes
// and its errors go to its own handler while it is still alive.
void sweep_retired() {
  std::vector<std::unique_ptr<bf::BfQueue>> due;
  {
    std::lock_guard<std::mutex> lock(retired_mutex());
    const auto now = Clock::now();
    auto &v = retired();
    for (size_t i = 0; i < v.size();) {
      if (now - v[i].at >= grace()) {
        due.push_back(std::move(v[i].q));
        v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
      } else {
        i++;
      }
    }
  }
  for (auto &q : due) {
    try {
      q->q.wait_and_throw();
    } catch (...) {
    }
    q.reset();
  }
}
} // namespace

namespace {
oneapi::mkl::transpose T(int32_t t) {
  switch (t) {
  case 1: return oneapi::mkl::transpose::trans;
  case 2: return oneapi::mkl::transpose::conjtrans;
  default: return oneapi::mkl::transpose::nontrans;
  }
}
} // namespace

// kind: 0 default selector, 1 cpu, 2 gpu. Returns null on failure (see bf_last_error).
// Every queue is created with the in_order property and an async_handler that records into the
// queue's own error store. In-order is NOT relied on for ordering: measured, oneMKL `dot` ignores
// it, so asynchronous submissions are also chained explicitly (common.hpp, `deps_chained`).
// See docs/ASYNC-DESIGN.md sections 3.4 and 5.
BF_API void *bf_queue_new(int32_t kind) {
  sweep_retired();
  void *out = nullptr;
  guard([&] {
    const sycl::property_list props{sycl::property::queue::in_order()};
    auto store = std::make_shared<bf::ErrStore>();
    switch (kind) {
    case 1: out = new bf::BfQueue{sycl::cpu_selector_v, store, props}; break;
    case 2: out = new bf::BfQueue{sycl::gpu_selector_v, store, props}; break;
    default: out = new bf::BfQueue{sycl::default_selector_v, store, props}; break;
    }
  });
  return out;
}

// The queue is retired and destroyed after a grace period (see `Retired` above). Errors nobody
// collected die with it: the store belongs to the queue.
BF_API void bf_queue_free(void *q) {
  if (!q) return;
  {
    std::lock_guard<std::mutex> lock(retired_mutex());
    retired().push_back(Retired{std::unique_ptr<bf::BfQueue>(static_cast<bf::BfQueue *>(q)), Clock::now()});
  }
  sweep_retired();
}

BF_API int32_t bf_queue_is_in_order(void *q) {
  int32_t r = -1;
  guard([&] { r = Q(q).is_in_order() ? 1 : 0; });
  return r;
}

// Wait for everything submitted to the queue; returns 1 if that surfaced an asynchronous error
// (this queue's, whichever thread SYCL ran the handler on).
BF_API int32_t bf_queue_wait(void *q) {
  int32_t rc = guard([&] { Q(q).wait_and_throw(); });
  int32_t arc = bf::drain_async_errors(*bf::Store(q));
  return rc != 0 ? rc : arc;
}

// Events are heap-allocated handles (the sycl::event plus its queue's error store) returned by the
// *_async functions.
BF_API int32_t bf_event_wait(void *e) {
  if (!e) {
    bf::g_last_error = "null event";
    return 3;
  }
  int32_t rc = guard([&] { bf::E(e).wait_and_throw(); });
  int32_t arc = bf::drain_async_errors(*static_cast<bf::BfEvent *>(e)->store);
  return rc != 0 ? rc : arc;
}

BF_API void bf_event_free(void *e) { delete static_cast<bf::BfEvent *>(e); }

// Diagnostic for the tests: submit a host task that throws, which SYCL reports asynchronously.
// Returns the event, or null on failure. Lets a test prove an asynchronous failure arrives as a
// status code rather than aborting the process.
BF_API void *bf_test_async_failure(void *q) {
  void *out = nullptr;
  guard([&] {
    out = new bf::BfEvent{Q(q).submit([&](sycl::handler &h) {
                            h.host_task([] {
                              throw std::runtime_error("bonefish test: forced asynchronous failure");
                            });
                          }),
                          bf::Store(q)};
  });
  return out;
}

// Copies the device name (NUL-terminated, truncated to cap) into buf; returns the full length.
BF_API int64_t bf_queue_device_name(void *q, char *buf, int64_t cap) {
  std::string n;
  if (guard([&] { n = Q(q).get_device().get_info<sycl::info::device::name>(); }) != 0) return -1;
  if (buf && cap > 0) {
    std::strncpy(buf, n.c_str(), static_cast<size_t>(cap - 1));
    buf[cap - 1] = '\0';
  }
  return static_cast<int64_t>(n.size());
}

// kind: 0 shared, 1 device, 2 host. Returns null on failure.
BF_API void *bf_malloc(void *q, int32_t kind, int64_t bytes) {
  void *p = nullptr;
  guard([&] {
    switch (kind) {
    case 1: p = sycl::malloc_device(static_cast<size_t>(bytes), Q(q)); break;
    case 2: p = sycl::malloc_host(static_cast<size_t>(bytes), Q(q)); break;
    default: p = sycl::malloc_shared(static_cast<size_t>(bytes), Q(q)); break;
    }
  });
  return p;
}

BF_API void bf_free(void *q, void *p) { sycl::free(p, Q(q)); }

BF_API int32_t bf_memcpy(void *q, void *dst, const void *src, int64_t bytes) {
  int32_t rc = guard([&] { Q(q).memcpy(dst, src, static_cast<size_t>(bytes)).wait(); });
  return rc != 0 ? rc : bf::settle(Q(q), *bf::Store(q));
}

// Column-major GEMM, blocking. transa/transb: 0 N, 1 T, 2 C.
#define BF_GEMM(NAME, TY)                                                                         \
  BF_API int32_t NAME(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,          \
                      TY alpha, const TY *a, int64_t lda, const TY *b, int64_t ldb, TY beta,     \
                      TY *c, int64_t ldc) {                                                      \
    return bf::guard_blocking(Q(q), *bf::Store(q), [&] {                                         \
      oneapi::mkl::blas::column_major::gemm(Q(q), T(ta), T(tb), m, n, k, alpha, a, lda, b, ldb,  \
                                            beta, c, ldc)                                        \
          .wait();                                                                               \
    });                                                                                          \
  }
BF_GEMM(bf_sgemm, float)
BF_GEMM(bf_dgemm, double)

// Non-blocking GEMM: returns the event (null on failure). Only the submission happens under the
// oneMKL lock; the device work runs after the call returns, so the caller must keep the operands
// alive and untouched until the event or the queue has been waited on. `deps` is an array of
// `n_deps` event handles the work must wait for (may be null with n_deps 0).
namespace {
template <class TY>
void *gemm_async(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k, TY alpha,
                 const TY *a, int64_t lda, const TY *b, int64_t ldb, TY beta, TY *c, int64_t ldc,
                 void *const *deps, int64_t n_deps) {
  void *out = nullptr;
  bf::guard_mkl([&] {
    out = bf::new_event(oneapi::mkl::blas::column_major::gemm(
                            Q(q), T(ta), T(tb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            bf::deps_chained(*bf::Store(q), deps, n_deps)),
                        bf::Store(q));
  });
  return out;
}
} // namespace

BF_API void *bf_sgemm_async(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,
                            float alpha, const float *a, int64_t lda, const float *b, int64_t ldb,
                            float beta, float *c, int64_t ldc, void *const *deps, int64_t n_deps) {
  return gemm_async(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps);
}

BF_API void *bf_dgemm_async(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,
                            double alpha, const double *a, int64_t lda, const double *b,
                            int64_t ldb, double beta, double *c, int64_t ldc, void *const *deps,
                            int64_t n_deps) {
  return gemm_async(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps);
}

// Diagnostic for tests and A/B runs: turn the asynchronous submission chain on (1) or off (0).
BF_API void bf_test_set_async_chain(int32_t on) { bf::async_chain_flag().store(on != 0); }

// Diagnostic: how many oneMKL calls have entered the lock path (see common.hpp). Lets a test prove
// that no oneMKL-calling function bypasses the lock.
BF_API int64_t bf_mkl_lock_acquisitions() { return bf::mkl_lock_acquisitions().load(); }

// Message of the last failure on this thread; valid until the next failing call on it.
BF_API const char *bf_last_error() { return bf::g_last_error.c_str(); }

// Device-side start and end of the work behind an event, in nanoseconds on the device's clock
// (SYCL command_start / command_end). Waits for the event first. Needs a queue created with
// profiling (bf_queue_new_ex flag 1): otherwise SYCL throws and this returns status 1 with its
// message. Returns 0 and fills `start_ns` / `end_ns` on success.
BF_API int32_t bf_event_profiling(void *e, uint64_t *start_ns, uint64_t *end_ns) {
  if (!e || !start_ns || !end_ns) {
    bf::g_last_error = "null argument";
    return 3;
  }
  return guard([&] {
    bf::E(e).wait();
    *start_ns = bf::E(e).get_profiling_info<sycl::info::event_profiling::command_start>();
    *end_ns = bf::E(e).get_profiling_info<sycl::info::event_profiling::command_end>();
  });
}

// An empty kernel, submitted like the other asynchronous calls (chained to the queue's previous
// one, `deps` as in bf_sgemm_async). Its device time is the launch floor to subtract from, or
// compare against, a real operation's. Null on failure.
BF_API void *bf_noop_async(void *q, void *const *deps, int64_t n_deps) {
  void *out = nullptr;
  guard([&] {
    auto d = bf::deps_chained(*bf::Store(q), deps, n_deps);
    sycl::event ev = Q(q).submit([&](sycl::handler &h) {
      h.depends_on(d);
      h.single_task([] {});
    });
    out = bf::new_event(ev, bf::Store(q));
  });
  return out;
}
