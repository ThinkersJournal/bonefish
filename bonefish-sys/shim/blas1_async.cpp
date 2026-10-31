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
