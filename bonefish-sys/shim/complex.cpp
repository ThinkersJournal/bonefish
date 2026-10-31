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
