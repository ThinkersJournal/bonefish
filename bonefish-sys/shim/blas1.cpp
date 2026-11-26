// SPDX-License-Identifier: MIT OR Apache-2.0
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
oneapi::mkl::uplo U(int32_t u) {
  return u == 0 ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
}
oneapi::mkl::side SD(int32_t s) {
  return s == 0 ? oneapi::mkl::side::left : oneapi::mkl::side::right;
}
oneapi::mkl::diag D(int32_t d) {
  return d == 0 ? oneapi::mkl::diag::nonunit : oneapi::mkl::diag::unit;
}
} // namespace

#define BF_BLAS1(P, TY)                                                                           \
  BF_API int32_t bf_##P##axpy(void *q, int64_t n, TY alpha, const TY *x, int64_t incx, TY *y,    \
                              int64_t incy) {                                                     \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] { blas::axpy(bf::Q(q), n, alpha, x, incx, y, incy).wait(); });           \
  }                                                                                               \
  BF_API int32_t bf_##P##scal(void *q, int64_t n, TY alpha, TY *x, int64_t incx) {               \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] { blas::scal(bf::Q(q), n, alpha, x, incx).wait(); });                    \
  }                                                                                               \
  BF_API int32_t bf_##P##copy(void *q, int64_t n, const TY *x, int64_t incx, TY *y,              \
                              int64_t incy) {                                                     \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] { blas::copy(bf::Q(q), n, x, incx, y, incy).wait(); });                  \
  }                                                                                               \
  BF_API int32_t bf_##P##swap(void *q, int64_t n, TY *x, int64_t incx, TY *y, int64_t incy) {    \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] { blas::swap(bf::Q(q), n, x, incx, y, incy).wait(); });                  \
  }                                                                                               \
  BF_API int32_t bf_##P##dot(void *q, int64_t n, const TY *x, int64_t incx, const TY *y,         \
                             int64_t incy, TY *result) {                                          \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] { blas::dot(bf::Q(q), n, x, incx, y, incy, result).wait(); });           \
  }                                                                                               \
  BF_API int32_t bf_##P##nrm2(void *q, int64_t n, const TY *x, int64_t incx, TY *result) {       \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] { blas::nrm2(bf::Q(q), n, x, incx, result).wait(); });                   \
  }                                                                                               \
  BF_API int32_t bf_##P##asum(void *q, int64_t n, const TY *x, int64_t incx, TY *result) {       \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] { blas::asum(bf::Q(q), n, x, incx, result).wait(); });                   \
  }                                                                                               \
  /* Zero-based index of the first element of largest magnitude. */                              \
  BF_API int32_t bf_##P##iamax(void *q, int64_t n, const TY *x, int64_t incx, int64_t *result) { \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {                                                                        \
      blas::iamax(bf::Q(q), n, x, incx, result, oneapi::mkl::index_base::zero).wait();            \
    });                                                                                           \
  }                                                                                               \
  /* Zero-based index of the first element of smallest magnitude. */                             \
  BF_API int32_t bf_##P##iamin(void *q, int64_t n, const TY *x, int64_t incx, int64_t *result) { \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {                                      \
      blas::iamin(bf::Q(q), n, x, incx, result, oneapi::mkl::index_base::zero).wait();            \
    });                                                                                           \
  }                                                                                               \
  /* Givens rotation of (x, y) by (c, s). */                                                      \
  BF_API int32_t bf_##P##rot(void *q, int64_t n, TY *x, int64_t incx, TY *y, int64_t incy, TY c,  \
                             TY s) {                                                              \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {                                      \
      blas::rot(bf::Q(q), n, x, incx, y, incy, c, s).wait();                                      \
    });                                                                                           \
  }                                                                                               \
  /* A = alpha * x * y^T + A; A is m x n, column-major. */                                        \
  BF_API int32_t bf_##P##ger(void *q, int64_t m, int64_t n, TY alpha, const TY *x, int64_t incx,  \
                             const TY *y, int64_t incy, TY *a, int64_t lda) {                     \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {                                      \
      blas::ger(bf::Q(q), m, n, alpha, x, incx, y, incy, a, lda).wait();                          \
    });                                                                                           \
  }                                                                                               \
  /* y = alpha * op(A) * x + beta * y; A is m x n, column-major. */                              \
  BF_API int32_t bf_##P##gemv(void *q, int32_t trans, int64_t m, int64_t n, TY alpha,            \
                              const TY *a, int64_t lda, const TY *x, int64_t incx, TY beta,      \
                              TY *y, int64_t incy) {                                              \
    return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {                                                                        \
      blas::gemv(bf::Q(q), T(trans), m, n, alpha, a, lda, x, incx, beta, y, incy).wait();        \
    });                                                                                           \
  }

BF_BLAS1(s, float)
BF_BLAS1(d, double)

// R2 remainder (rotm, symv, syr, syr2, trmv, trsv, gbmv). uplo: 0 upper, 1 lower; diag: 0 non-unit,
// 1 unit; trans: 0 N, 1 T, 2 C.
BF_API int32_t bf_srotm(void *q, int64_t n, float * x, int64_t incx, float * y, int64_t incy, float * param) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::rotm(bf::Q(q), n, x, incx, y, incy, param).wait();
  });
}

BF_API int32_t bf_ssymv(void *q, int32_t uplo, int64_t n, float alpha, const float * a, int64_t lda, const float * x, int64_t incx, float beta, float * y, int64_t incy) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::symv(bf::Q(q), U(uplo), n, alpha, a, lda, x, incx, beta, y, incy).wait();
  });
}

BF_API int32_t bf_ssyr(void *q, int32_t uplo, int64_t n, float alpha, const float * x, int64_t incx, float * a, int64_t lda) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syr(bf::Q(q), U(uplo), n, alpha, x, incx, a, lda).wait();
  });
}

BF_API int32_t bf_ssyr2(void *q, int32_t uplo, int64_t n, float alpha, const float * x, int64_t incx, const float * y, int64_t incy, float * a, int64_t lda) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syr2(bf::Q(q), U(uplo), n, alpha, x, incx, y, incy, a, lda).wait();
  });
}

BF_API int32_t bf_strmv(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const float * a, int64_t lda, float * x, int64_t incx) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trmv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx).wait();
  });
}

BF_API int32_t bf_strsv(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const float * a, int64_t lda, float * x, int64_t incx) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trsv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx).wait();
  });
}

BF_API int32_t bf_sgbmv(void *q, int32_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, float alpha, const float * a, int64_t lda, const float * x, int64_t incx, float beta, float * y, int64_t incy) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::gbmv(bf::Q(q), T(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy).wait();
  });
}

BF_API int32_t bf_drotm(void *q, int64_t n, double * x, int64_t incx, double * y, int64_t incy, double * param) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::rotm(bf::Q(q), n, x, incx, y, incy, param).wait();
  });
}

BF_API int32_t bf_dsymv(void *q, int32_t uplo, int64_t n, double alpha, const double * a, int64_t lda, const double * x, int64_t incx, double beta, double * y, int64_t incy) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::symv(bf::Q(q), U(uplo), n, alpha, a, lda, x, incx, beta, y, incy).wait();
  });
}

BF_API int32_t bf_dsyr(void *q, int32_t uplo, int64_t n, double alpha, const double * x, int64_t incx, double * a, int64_t lda) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syr(bf::Q(q), U(uplo), n, alpha, x, incx, a, lda).wait();
  });
}

BF_API int32_t bf_dsyr2(void *q, int32_t uplo, int64_t n, double alpha, const double * x, int64_t incx, const double * y, int64_t incy, double * a, int64_t lda) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syr2(bf::Q(q), U(uplo), n, alpha, x, incx, y, incy, a, lda).wait();
  });
}

BF_API int32_t bf_dtrmv(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const double * a, int64_t lda, double * x, int64_t incx) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trmv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx).wait();
  });
}

BF_API int32_t bf_dtrsv(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const double * a, int64_t lda, double * x, int64_t incx) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trsv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx).wait();
  });
}

BF_API int32_t bf_dgbmv(void *q, int32_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, double alpha, const double * a, int64_t lda, const double * x, int64_t incx, double beta, double * y, int64_t incy) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::gbmv(bf::Q(q), T(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy).wait();
  });
}

// Level 3 (R2 remainder: symm, syrk, syr2k, trmm, trsm). side: 0 left, 1 right; uplo: 0 upper, 1 lower;
// diag: 0 non-unit, 1 unit; trans: 0 N, 1 T, 2 C.
BF_API int32_t bf_ssymm(void *q, int32_t side, int32_t uplo, int64_t m, int64_t n, float alpha, const float * a, int64_t lda, const float * b, int64_t ldb, float beta, float *  c, int64_t ldc) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::symm(bf::Q(q), SD(side), U(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc).wait();
  });
}

BF_API int32_t bf_ssyrk(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, float alpha, const float * a, int64_t lda, float beta, float *  c, int64_t ldc) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syrk(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, beta, c, ldc).wait();
  });
}

BF_API int32_t bf_ssyr2k(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, float alpha, const float * a, int64_t lda, const float * b, int64_t ldb, float beta, float *  c, int64_t ldc) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syr2k(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc).wait();
  });
}

BF_API int32_t bf_strmm(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, float alpha, const float * a, int64_t lda, float *  b, int64_t ldb) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trmm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb).wait();
  });
}

BF_API int32_t bf_strsm(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, float alpha, const float * a, int64_t lda, float *  b, int64_t ldb) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trsm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb).wait();
  });
}

BF_API int32_t bf_dsymm(void *q, int32_t side, int32_t uplo, int64_t m, int64_t n, double alpha, const double * a, int64_t lda, const double * b, int64_t ldb, double beta, double *  c, int64_t ldc) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::symm(bf::Q(q), SD(side), U(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc).wait();
  });
}

BF_API int32_t bf_dsyrk(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, double alpha, const double * a, int64_t lda, double beta, double *  c, int64_t ldc) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syrk(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, beta, c, ldc).wait();
  });
}

BF_API int32_t bf_dsyr2k(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, double alpha, const double * a, int64_t lda, const double * b, int64_t ldb, double beta, double *  c, int64_t ldc) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::syr2k(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc).wait();
  });
}

BF_API int32_t bf_dtrmm(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, double alpha, const double * a, int64_t lda, double *  b, int64_t ldb) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trmm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb).wait();
  });
}

BF_API int32_t bf_dtrsm(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, double alpha, const double * a, int64_t lda, double *  b, int64_t ldb) {
  return bf::guard_blocking(bf::Q(q), *bf::Store(q), [&] {
    blas::trsm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb).wait();
  });
}
