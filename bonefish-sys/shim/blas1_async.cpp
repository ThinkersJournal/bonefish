// Non-blocking BLAS level 1 and gemv (single and double): the same operations as blas1.cpp, but
// each returns the sycl::event (null on failure, see bf_last_error) instead of waiting. Only the
// submission runs under the oneMKL lock. The caller keeps every operand alive, and untouched by
// the host, until the event or the queue has been waited on; scalar results (dot, nrm2, asum,
// iamax) are written through the USM `result` pointer when the work runs, not when this returns.
// `deps` points at `n_deps` event handles the work waits for (null with 0 for none).
//
// Generated once by a script from the list of operations; edit by hand from here.
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

// Submit under the oneMKL lock and wrap the event with the queue's error store.
template <class F> void *submit(void *q, F &&f) {
  void *out = nullptr;
  bf::guard_mkl([&] { out = bf::new_event(f(), bf::Store(q)); });
  return out;
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

BF_API void *bf_saxpy_async(void *q, int64_t n, float alpha, const float *x, int64_t incx, float *y,
                           int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::axpy(bf::Q(q), n, alpha, x, incx, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_sscal_async(void *q, int64_t n, float alpha, float *x, int64_t incx, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::scal(bf::Q(q), n, alpha, x, incx, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_scopy_async(void *q, int64_t n, const float *x, int64_t incx, float *y, int64_t incy,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::copy(bf::Q(q), n, x, incx, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_sswap_async(void *q, int64_t n, float *x, int64_t incx, float *y, int64_t incy,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::swap(bf::Q(q), n, x, incx, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_sdot_async(void *q, int64_t n, const float *x, int64_t incx, const float *y,
                          int64_t incy, float *result, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::dot(bf::Q(q), n, x, incx, y, incy, result, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_snrm2_async(void *q, int64_t n, const float *x, int64_t incx, float *result,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::nrm2(bf::Q(q), n, x, incx, result, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_sasum_async(void *q, int64_t n, const float *x, int64_t incx, float *result,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::asum(bf::Q(q), n, x, incx, result, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

/* Zero-based index of the first element of largest magnitude. */
BF_API void *bf_siamax_async(void *q, int64_t n, const float *x, int64_t incx, int64_t *result,
                            void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::iamax(bf::Q(q), n, x, incx, result, oneapi::mkl::index_base::zero, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

/* y = alpha * op(A) * x + beta * y; A is m x n, column-major. */
BF_API void *bf_sgemv_async(void *q, int32_t trans, int64_t m, int64_t n, float alpha,
                           const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                           float *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::gemv(bf::Q(q), T(trans), m, n, alpha, a, lda, x, incx, beta, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_daxpy_async(void *q, int64_t n, double alpha, const double *x, int64_t incx, double *y,
                           int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::axpy(bf::Q(q), n, alpha, x, incx, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_dscal_async(void *q, int64_t n, double alpha, double *x, int64_t incx, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::scal(bf::Q(q), n, alpha, x, incx, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_dcopy_async(void *q, int64_t n, const double *x, int64_t incx, double *y, int64_t incy,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::copy(bf::Q(q), n, x, incx, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_dswap_async(void *q, int64_t n, double *x, int64_t incx, double *y, int64_t incy,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::swap(bf::Q(q), n, x, incx, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_ddot_async(void *q, int64_t n, const double *x, int64_t incx, const double *y,
                          int64_t incy, double *result, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::dot(bf::Q(q), n, x, incx, y, incy, result, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_dnrm2_async(void *q, int64_t n, const double *x, int64_t incx, double *result,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::nrm2(bf::Q(q), n, x, incx, result, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_dasum_async(void *q, int64_t n, const double *x, int64_t incx, double *result,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::asum(bf::Q(q), n, x, incx, result, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

/* Zero-based index of the first element of largest magnitude. */
BF_API void *bf_diamax_async(void *q, int64_t n, const double *x, int64_t incx, int64_t *result,
                            void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::iamax(bf::Q(q), n, x, incx, result, oneapi::mkl::index_base::zero, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

/* y = alpha * op(A) * x + beta * y; A is m x n, column-major. */
BF_API void *bf_dgemv_async(void *q, int32_t trans, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                           double *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::gemv(bf::Q(q), T(trans), m, n, alpha, a, lda, x, incx, beta, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_siamin_async(void *q, int64_t n, const float *x, int64_t incx, int64_t *result,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::iamin(bf::Q(q), n, x, incx, result, oneapi::mkl::index_base::zero, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_srot_async(void *q, int64_t n, float *x, int64_t incx, float *y, int64_t incy, float c,
                           float s, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::rot(bf::Q(q), n, x, incx, y, incy, c, s, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_sger_async(void *q, int64_t m, int64_t n, float alpha, const float *x, int64_t incx,
                           const float *y, int64_t incy, float *a, int64_t lda, void *const *deps,
                           int64_t n_deps) {
  return submit(q, [&] { return blas::ger(bf::Q(q), m, n, alpha, x, incx, y, incy, a, lda, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_diamin_async(void *q, int64_t n, const double *x, int64_t incx, int64_t *result,
                           void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::iamin(bf::Q(q), n, x, incx, result, oneapi::mkl::index_base::zero, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_drot_async(void *q, int64_t n, double *x, int64_t incx, double *y, int64_t incy, double c,
                           double s, void *const *deps, int64_t n_deps) {
  return submit(q, [&] { return blas::rot(bf::Q(q), n, x, incx, y, incy, c, s, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

BF_API void *bf_dger_async(void *q, int64_t m, int64_t n, double alpha, const double *x, int64_t incx,
                           const double *y, int64_t incy, double *a, int64_t lda, void *const *deps,
                           int64_t n_deps) {
  return submit(q, [&] { return blas::ger(bf::Q(q), m, n, alpha, x, incx, y, incy, a, lda, bf::deps_chained(*bf::Store(q), deps, n_deps)); });
}

// R2 remainder (rotm, symv, syr, syr2, trmv, trsv, gbmv); uplo: 0 upper, 1 lower; diag: 0 non-unit, 1 unit.
BF_API void *bf_srotm_async(void *q, int64_t n, float * x, int64_t incx, float * y, int64_t incy, float * param, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::rotm(bf::Q(q), n, x, incx, y, incy, param, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_ssymv_async(void *q, int32_t uplo, int64_t n, float alpha, const float * a, int64_t lda, const float * x, int64_t incx, float beta, float * y, int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::symv(bf::Q(q), U(uplo), n, alpha, a, lda, x, incx, beta, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_ssyr_async(void *q, int32_t uplo, int64_t n, float alpha, const float * x, int64_t incx, float * a, int64_t lda, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syr(bf::Q(q), U(uplo), n, alpha, x, incx, a, lda, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_ssyr2_async(void *q, int32_t uplo, int64_t n, float alpha, const float * x, int64_t incx, const float * y, int64_t incy, float * a, int64_t lda, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syr2(bf::Q(q), U(uplo), n, alpha, x, incx, y, incy, a, lda, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_strmv_async(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const float * a, int64_t lda, float * x, int64_t incx, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trmv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_strsv_async(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const float * a, int64_t lda, float * x, int64_t incx, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trsv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_sgbmv_async(void *q, int32_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, float alpha, const float * a, int64_t lda, const float * x, int64_t incx, float beta, float * y, int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::gbmv(bf::Q(q), T(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_drotm_async(void *q, int64_t n, double * x, int64_t incx, double * y, int64_t incy, double * param, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::rotm(bf::Q(q), n, x, incx, y, incy, param, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dsymv_async(void *q, int32_t uplo, int64_t n, double alpha, const double * a, int64_t lda, const double * x, int64_t incx, double beta, double * y, int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::symv(bf::Q(q), U(uplo), n, alpha, a, lda, x, incx, beta, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dsyr_async(void *q, int32_t uplo, int64_t n, double alpha, const double * x, int64_t incx, double * a, int64_t lda, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syr(bf::Q(q), U(uplo), n, alpha, x, incx, a, lda, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dsyr2_async(void *q, int32_t uplo, int64_t n, double alpha, const double * x, int64_t incx, const double * y, int64_t incy, double * a, int64_t lda, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syr2(bf::Q(q), U(uplo), n, alpha, x, incx, y, incy, a, lda, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dtrmv_async(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const double * a, int64_t lda, double * x, int64_t incx, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trmv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dtrsv_async(void *q, int32_t uplo, int32_t trans, int32_t diag, int64_t n, const double * a, int64_t lda, double * x, int64_t incx, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trsv(bf::Q(q), U(uplo), T(trans), D(diag), n, a, lda, x, incx, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dgbmv_async(void *q, int32_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, double alpha, const double * a, int64_t lda, const double * x, int64_t incx, double beta, double * y, int64_t incy, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::gbmv(bf::Q(q), T(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

// Level 3 (R2 remainder: symm, syrk, syr2k, trmm, trsm). side: 0 left, 1 right; uplo: 0 upper, 1 lower;
// diag: 0 non-unit, 1 unit; trans: 0 N, 1 T, 2 C.
BF_API void *bf_ssymm_async(void *q, int32_t side, int32_t uplo, int64_t m, int64_t n, float alpha, const float * a, int64_t lda, const float * b, int64_t ldb, float beta, float *  c, int64_t ldc, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::symm(bf::Q(q), SD(side), U(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_ssyrk_async(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, float alpha, const float * a, int64_t lda, float beta, float *  c, int64_t ldc, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syrk(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, beta, c, ldc, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_ssyr2k_async(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, float alpha, const float * a, int64_t lda, const float * b, int64_t ldb, float beta, float *  c, int64_t ldc, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syr2k(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_strmm_async(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, float alpha, const float * a, int64_t lda, float *  b, int64_t ldb, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trmm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_strsm_async(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, float alpha, const float * a, int64_t lda, float *  b, int64_t ldb, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trsm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dsymm_async(void *q, int32_t side, int32_t uplo, int64_t m, int64_t n, double alpha, const double * a, int64_t lda, const double * b, int64_t ldb, double beta, double *  c, int64_t ldc, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::symm(bf::Q(q), SD(side), U(uplo), m, n, alpha, a, lda, b, ldb, beta, c, ldc, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dsyrk_async(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, double alpha, const double * a, int64_t lda, double beta, double *  c, int64_t ldc, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syrk(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, beta, c, ldc, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dsyr2k_async(void *q, int32_t uplo, int32_t trans, int64_t n, int64_t k, double alpha, const double * a, int64_t lda, const double * b, int64_t ldb, double beta, double *  c, int64_t ldc, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::syr2k(bf::Q(q), U(uplo), T(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dtrmm_async(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, double alpha, const double * a, int64_t lda, double *  b, int64_t ldb, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trmm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}

BF_API void *bf_dtrsm_async(void *q, int32_t side, int32_t uplo, int32_t trans, int32_t diag, int64_t m, int64_t n, double alpha, const double * a, int64_t lda, double *  b, int64_t ldb, void *const *deps, int64_t n_deps) {
  return submit(q, [&] {
    return blas::trsm(bf::Q(q), SD(side), U(uplo), T(trans), D(diag), m, n, alpha, a, lda, b, ldb, bf::deps_chained(*bf::Store(q), deps, n_deps));
  });
}
