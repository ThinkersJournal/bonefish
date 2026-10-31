// Complex BLAS over oneapi::mkl::blas::column_major (single = `c`, double = `z`).
//
// Complex data crosses the C boundary as interleaved real/imaginary floats (or doubles), the
// layout of std::complex, and the complex scalars (alpha, beta) are passed BY POINTER to two reals,
// never by value: passing a C++ class like std::complex by value through extern "C" is not
// something to rely on across compilers and ABIs. Everything else matches the real functions:
// column-major, blocking functions wait, `*_async` functions return the event (null on failure)
// chained to the queue's previous asynchronous call, and every oneMKL entry takes the oneMKL lock.
#include "common.hpp"

#include <complex>

namespace blas = oneapi::mkl::blas::column_major;

namespace {
oneapi::mkl::transpose T(int32_t t) {
  switch (t) {
  case 1: return oneapi::mkl::transpose::trans;
  case 2: return oneapi::mkl::transpose::conjtrans;
  default: return oneapi::mkl::transpose::nontrans;
  }
}

template <class R> using Cx = std::complex<R>;
template <class R> const Cx<R> *cx(const R *p) { return reinterpret_cast<const Cx<R> *>(p); }
template <class R> Cx<R> *cx(R *p) { return reinterpret_cast<Cx<R> *>(p); }

// C = alpha * op(A) * op(B) + beta * C.
template <class R>
int32_t gemm_blocking(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,
                      const R *alpha, const R *a, int64_t lda, const R *b, int64_t ldb,
                      const R *beta, R *c, int64_t ldc) {
  return bf::guard_mkl([&] {
    const Cx<R> al(alpha[0], alpha[1]);
    const Cx<R> be(beta[0], beta[1]);
    blas::gemm(bf::Q(q), T(ta), T(tb), m, n, k, al, cx(a), lda, cx(b), ldb, be, cx(c), ldc).wait();
  });
}

template <class R>
void *gemm_async(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k, const R *alpha,
                 const R *a, int64_t lda, const R *b, int64_t ldb, const R *beta, R *c,
                 int64_t ldc, void *const *deps, int64_t n_deps) {
  void *out = nullptr;
  bf::guard_mkl([&] {
    const Cx<R> al(alpha[0], alpha[1]);
    const Cx<R> be(beta[0], beta[1]);
    out = bf::new_event(blas::gemm(bf::Q(q), T(ta), T(tb), m, n, k, al, cx(a), lda, cx(b), ldb, be,
                                   cx(c), ldc, bf::deps_chained(*bf::Store(q), deps, n_deps)),
                        bf::Store(q));
  });
  return out;
}
} // namespace

BF_API int32_t bf_cgemm(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,
                        const float *alpha, const float *a, int64_t lda, const float *b,
                        int64_t ldb, const float *beta, float *c, int64_t ldc) {
  return gemm_blocking<float>(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

BF_API int32_t bf_zgemm(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,
                        const double *alpha, const double *a, int64_t lda, const double *b,
                        int64_t ldb, const double *beta, double *c, int64_t ldc) {
  return gemm_blocking<double>(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

BF_API void *bf_cgemm_async(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,
                            const float *alpha, const float *a, int64_t lda, const float *b,
                            int64_t ldb, const float *beta, float *c, int64_t ldc,
                            void *const *deps, int64_t n_deps) {
  return gemm_async<float>(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps);
}

BF_API void *bf_zgemm_async(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,
                            const double *alpha, const double *a, int64_t lda, const double *b,
                            int64_t ldb, const double *beta, double *c, int64_t ldc,
                            void *const *deps, int64_t n_deps) {
  return gemm_async<double>(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps, n_deps);
}

// ---- Level 1 and gemv, non-blocking only. The safe blocking functions are built on a scope, so no
// blocking C entry points are needed. Each call returns the event (null on failure), is chained to
// the queue's previous asynchronous call, and takes the oneMKL lock for the submission. Complex
// scalars (alpha, beta) come in by pointer to two reals; a complex RESULT (dotu, dotc) is written
// through a pointer to two reals; nrm2 and asum of complex data produce a REAL result.
namespace {
template <class F> void *submit(void *q, F &&f) {
  void *out = nullptr;
  bf::guard_mkl([&] { out = bf::new_event(f(), bf::Store(q)); });
  return out;
}
template <class R> Cx<R> sc(const R *p) { return Cx<R>(p[0], p[1]); }
template <class R> void *axpy_a(void *q, int64_t n, const R *alpha, const R *x, int64_t incx, R *y, int64_t incy, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::axpy(bf::Q(q), n, sc(alpha), cx(x), incx, cx(y), incy, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *scal_a(void *q, int64_t n, const R *alpha, R *x, int64_t incx, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::scal(bf::Q(q), n, sc(alpha), cx(x), incx, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *copy_a(void *q, int64_t n, const R *x, int64_t incx, R *y, int64_t incy, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::copy(bf::Q(q), n, cx(x), incx, cx(y), incy, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *swap_a(void *q, int64_t n, R *x, int64_t incx, R *y, int64_t incy, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::swap(bf::Q(q), n, cx(x), incx, cx(y), incy, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *dotu_a(void *q, int64_t n, const R *x, int64_t incx, const R *y, int64_t incy, R *result, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::dotu(bf::Q(q), n, cx(x), incx, cx(y), incy, cx(result), bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *dotc_a(void *q, int64_t n, const R *x, int64_t incx, const R *y, int64_t incy, R *result, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::dotc(bf::Q(q), n, cx(x), incx, cx(y), incy, cx(result), bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *nrm2_a(void *q, int64_t n, const R *x, int64_t incx, R *result, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::nrm2(bf::Q(q), n, cx(x), incx, result, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *asum_a(void *q, int64_t n, const R *x, int64_t incx, R *result, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::asum(bf::Q(q), n, cx(x), incx, result, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *iamax_a(void *q, int64_t n, const R *x, int64_t incx, int64_t *result, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::iamax(bf::Q(q), n, cx(x), incx, result, oneapi::mkl::index_base::zero, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
template <class R> void *gemv_a(void *q, int32_t trans, int64_t m, int64_t n, const R *alpha, const R *a, int64_t lda, const R *x, int64_t incx, const R *beta, R *y, int64_t incy, void *const *deps, int64_t nd) {
  return submit(q, [&] { return blas::gemv(bf::Q(q), T(trans), m, n, sc(alpha), cx(a), lda, cx(x), incx, sc(beta), cx(y), incy, bf::deps_chained(*bf::Store(q), deps, nd)); });
}
} // namespace

BF_API void *bf_caxpy_async(void *q, int64_t n, const float *alpha, const float *x, int64_t incx, float *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return axpy_a<float>(q, n, alpha, x, incx, y, incy, deps, n_deps);
}

BF_API void *bf_cscal_async(void *q, int64_t n, const float *alpha, float *x, int64_t incx, void *const *deps, int64_t n_deps) {
  return scal_a<float>(q, n, alpha, x, incx, deps, n_deps);
}

BF_API void *bf_ccopy_async(void *q, int64_t n, const float *x, int64_t incx, float *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return copy_a<float>(q, n, x, incx, y, incy, deps, n_deps);
}

BF_API void *bf_cswap_async(void *q, int64_t n, float *x, int64_t incx, float *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return swap_a<float>(q, n, x, incx, y, incy, deps, n_deps);
}

BF_API void *bf_cdotu_async(void *q, int64_t n, const float *x, int64_t incx, const float *y, int64_t incy, float *result, void *const *deps, int64_t n_deps) {
  return dotu_a<float>(q, n, x, incx, y, incy, result, deps, n_deps);
}

BF_API void *bf_cdotc_async(void *q, int64_t n, const float *x, int64_t incx, const float *y, int64_t incy, float *result, void *const *deps, int64_t n_deps) {
  return dotc_a<float>(q, n, x, incx, y, incy, result, deps, n_deps);
}

BF_API void *bf_cnrm2_async(void *q, int64_t n, const float *x, int64_t incx, float *result, void *const *deps, int64_t n_deps) {
  return nrm2_a<float>(q, n, x, incx, result, deps, n_deps);
}

BF_API void *bf_casum_async(void *q, int64_t n, const float *x, int64_t incx, float *result, void *const *deps, int64_t n_deps) {
  return asum_a<float>(q, n, x, incx, result, deps, n_deps);
}

BF_API void *bf_ciamax_async(void *q, int64_t n, const float *x, int64_t incx, int64_t *result, void *const *deps, int64_t n_deps) {
  return iamax_a<float>(q, n, x, incx, result, deps, n_deps);
}

BF_API void *bf_cgemv_async(void *q, int32_t trans, int64_t m, int64_t n, const float *alpha, const float *a, int64_t lda, const float *x, int64_t incx, const float *beta, float *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return gemv_a<float>(q, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, deps, n_deps);
}

BF_API void *bf_zaxpy_async(void *q, int64_t n, const double *alpha, const double *x, int64_t incx, double *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return axpy_a<double>(q, n, alpha, x, incx, y, incy, deps, n_deps);
}

BF_API void *bf_zscal_async(void *q, int64_t n, const double *alpha, double *x, int64_t incx, void *const *deps, int64_t n_deps) {
  return scal_a<double>(q, n, alpha, x, incx, deps, n_deps);
}

BF_API void *bf_zcopy_async(void *q, int64_t n, const double *x, int64_t incx, double *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return copy_a<double>(q, n, x, incx, y, incy, deps, n_deps);
}

BF_API void *bf_zswap_async(void *q, int64_t n, double *x, int64_t incx, double *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return swap_a<double>(q, n, x, incx, y, incy, deps, n_deps);
}

BF_API void *bf_zdotu_async(void *q, int64_t n, const double *x, int64_t incx, const double *y, int64_t incy, double *result, void *const *deps, int64_t n_deps) {
  return dotu_a<double>(q, n, x, incx, y, incy, result, deps, n_deps);
}

BF_API void *bf_zdotc_async(void *q, int64_t n, const double *x, int64_t incx, const double *y, int64_t incy, double *result, void *const *deps, int64_t n_deps) {
  return dotc_a<double>(q, n, x, incx, y, incy, result, deps, n_deps);
}

BF_API void *bf_znrm2_async(void *q, int64_t n, const double *x, int64_t incx, double *result, void *const *deps, int64_t n_deps) {
  return nrm2_a<double>(q, n, x, incx, result, deps, n_deps);
}

BF_API void *bf_zasum_async(void *q, int64_t n, const double *x, int64_t incx, double *result, void *const *deps, int64_t n_deps) {
  return asum_a<double>(q, n, x, incx, result, deps, n_deps);
}

BF_API void *bf_ziamax_async(void *q, int64_t n, const double *x, int64_t incx, int64_t *result, void *const *deps, int64_t n_deps) {
  return iamax_a<double>(q, n, x, incx, result, deps, n_deps);
}

BF_API void *bf_zgemv_async(void *q, int32_t trans, int64_t m, int64_t n, const double *alpha, const double *a, int64_t lda, const double *x, int64_t incx, const double *beta, double *y, int64_t incy, void *const *deps, int64_t n_deps) {
  return gemv_a<double>(q, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, deps, n_deps);
}
