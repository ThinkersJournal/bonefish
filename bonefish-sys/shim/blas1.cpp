// BLAS level 1 and gemv over oneapi::mkl::blas::column_major, single and double precision.
// Every call is blocking. Pointers are USM reachable from the queue; scalar results (dot, nrm2,
// asum, iamax) are written through a USM result pointer, as oneMKL requires.
#include "common.hpp"

namespace blas = oneapi::mkl::blas::column_major;

namespace {
oneapi::mkl::transpose T(int32_t t) {
  switch (t) {
  case 1: return oneapi::mkl::transpose::trans;
  case 2: return oneapi::mkl::transpose::conjtrans;
  default: return oneapi::mkl::transpose::nontrans;
  }
}
} // namespace

#define BF_BLAS1(P, TY)                                                                           \
  BF_API int32_t bf_##P##axpy(void *q, int64_t n, TY alpha, const TY *x, int64_t incx, TY *y,    \
                              int64_t incy) {                                                     \
    return bf::guard_mkl([&] { blas::axpy(bf::Q(q), n, alpha, x, incx, y, incy).wait(); });           \
  }                                                                                               \
  BF_API int32_t bf_##P##scal(void *q, int64_t n, TY alpha, TY *x, int64_t incx) {               \
    return bf::guard_mkl([&] { blas::scal(bf::Q(q), n, alpha, x, incx).wait(); });                    \
  }                                                                                               \
  BF_API int32_t bf_##P##copy(void *q, int64_t n, const TY *x, int64_t incx, TY *y,              \
                              int64_t incy) {                                                     \
    return bf::guard_mkl([&] { blas::copy(bf::Q(q), n, x, incx, y, incy).wait(); });                  \
  }                                                                                               \
  BF_API int32_t bf_##P##swap(void *q, int64_t n, TY *x, int64_t incx, TY *y, int64_t incy) {    \
    return bf::guard_mkl([&] { blas::swap(bf::Q(q), n, x, incx, y, incy).wait(); });                  \
  }                                                                                               \
  BF_API int32_t bf_##P##dot(void *q, int64_t n, const TY *x, int64_t incx, const TY *y,         \
                             int64_t incy, TY *result) {                                          \
    return bf::guard_mkl([&] { blas::dot(bf::Q(q), n, x, incx, y, incy, result).wait(); });           \
  }                                                                                               \
  BF_API int32_t bf_##P##nrm2(void *q, int64_t n, const TY *x, int64_t incx, TY *result) {       \
    return bf::guard_mkl([&] { blas::nrm2(bf::Q(q), n, x, incx, result).wait(); });                   \
  }                                                                                               \
  BF_API int32_t bf_##P##asum(void *q, int64_t n, const TY *x, int64_t incx, TY *result) {       \
    return bf::guard_mkl([&] { blas::asum(bf::Q(q), n, x, incx, result).wait(); });                   \
  }                                                                                               \
  /* Zero-based index of the first element of largest magnitude. */                              \
  BF_API int32_t bf_##P##iamax(void *q, int64_t n, const TY *x, int64_t incx, int64_t *result) { \
    return bf::guard_mkl([&] {                                                                        \
      blas::iamax(bf::Q(q), n, x, incx, result, oneapi::mkl::index_base::zero).wait();            \
    });                                                                                           \
  }                                                                                               \
  /* y = alpha * op(A) * x + beta * y; A is m x n, column-major. */                              \
  BF_API int32_t bf_##P##gemv(void *q, int32_t trans, int64_t m, int64_t n, TY alpha,            \
                              const TY *a, int64_t lda, const TY *x, int64_t incx, TY beta,      \
                              TY *y, int64_t incy) {                                              \
    return bf::guard_mkl([&] {                                                                        \
      blas::gemv(bf::Q(q), T(trans), m, n, alpha, a, lda, x, incx, beta, y, incy).wait();        \
    });                                                                                           \
  }

BF_BLAS1(s, float)
BF_BLAS1(d, double)
