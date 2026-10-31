// extern "C" shim over oneapi::mkl (SYCL). Opaque handles only; no C++ type crosses the boundary.
#include "common.hpp"

#include <cstring>

thread_local std::string bf::g_last_error;

using bf::guard;
using bf::Q;

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
// Every queue is in-order (work on one queue runs in submission order) and has an async_handler
// that records into the queue's own error store; both are required by the non-blocking API
// (docs/ASYNC-DESIGN.md sections 3.4 and 5).
BF_API void *bf_queue_new(int32_t kind) {
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

// Errors nobody collected die with the queue: the store belongs to it.
BF_API void bf_queue_free(void *q) { delete static_cast<bf::BfQueue *>(q); }

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
  return guard([&] { Q(q).memcpy(dst, src, static_cast<size_t>(bytes)).wait(); });
}

// Column-major GEMM, blocking. transa/transb: 0 N, 1 T, 2 C.
#define BF_GEMM(NAME, TY)                                                                         \
  BF_API int32_t NAME(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,          \
                      TY alpha, const TY *a, int64_t lda, const TY *b, int64_t ldb, TY beta,     \
                      TY *c, int64_t ldc) {                                                      \
    return bf::guard_mkl([&] {                                                                   \
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
    out = new bf::BfEvent{oneapi::mkl::blas::column_major::gemm(
                              Q(q), T(ta), T(tb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                              bf::deps_of(deps, n_deps)),
                          bf::Store(q)};
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

// Diagnostic: how many oneMKL calls have entered the lock path (see common.hpp). Lets a test prove
// that no oneMKL-calling function bypasses the lock.
BF_API int64_t bf_mkl_lock_acquisitions() { return bf::mkl_lock_acquisitions().load(); }

// Message of the last failure on this thread; valid until the next failing call on it.
BF_API const char *bf_last_error() { return bf::g_last_error.c_str(); }
