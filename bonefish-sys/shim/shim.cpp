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
BF_API void *bf_queue_new(int32_t kind) {
  void *out = nullptr;
  guard([&] {
    switch (kind) {
    case 1: out = new sycl::queue{sycl::cpu_selector_v}; break;
    case 2: out = new sycl::queue{sycl::gpu_selector_v}; break;
    default: out = new sycl::queue{sycl::default_selector_v}; break;
    }
  });
  return out;
}

BF_API void bf_queue_free(void *q) { delete static_cast<sycl::queue *>(q); }

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

// Diagnostic: how many oneMKL calls have entered the lock path (see common.hpp). Lets a test prove
// that no oneMKL-calling function bypasses the lock.
BF_API int64_t bf_mkl_lock_acquisitions() { return bf::mkl_lock_acquisitions().load(); }

// Message of the last failure on this thread; valid until the next failing call on it.
BF_API const char *bf_last_error() { return bf::g_last_error.c_str(); }
