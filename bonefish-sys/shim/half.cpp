// Half-precision `gemm` over oneapi::mkl::blas::column_major (R5c, non-blocking only).
//
// The four variants oneMKL declares (usm_decls.hpp), named by (A, B, C, scalar):
//   bf_hgemm_async      (half, half, half, half)       C is half; alpha/beta cross as float and are
//                                                      converted; internal accumulation unstated.
//   bf_hgemm_f32_async  (half, half, float, float)     f32 output and scalars.
//   bf_bgemm_async      (bf16, bf16, bf16, float)      C is bf16, f32 scalars.
//   bf_bgemm_f32_async  (bf16, bf16, float, float)     f32 output and scalars.
// Half and bf16 data cross the C boundary as raw 16-bit words (`uint16_t`), the bit layout of
// sycl::half and of bfloat16; the Rust side owns the conversion. Like every `*_async` function:
// returns the event (null on failure), chained to the queue's previous submission, under the
// oneMKL lock. Whether a given device runs these is not assumed: failure comes back as null with
// the message in bf_last_error.
#include "common.hpp"

namespace blas = oneapi::mkl::blas::column_major;

namespace {
oneapi::mkl::transpose T(int32_t t) {
  switch (t) {
  case 1:
  case 2: return oneapi::mkl::transpose::trans;
  default: return oneapi::mkl::transpose::nontrans;
  }
}

// TA: element type of A and B, TC: element type of C, TS: scalar type oneMKL takes, P: the raw
// 16-bit pointer type the C boundary uses for A/B (and for C when it is 16-bit).
template <class TA, class TC, class TS>
void *hgemm_async(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k, float alpha,
                  const uint16_t *a, int64_t lda, const uint16_t *b, int64_t ldb, float beta,
                  void *c, int64_t ldc, void *const *deps, int64_t n_deps) {
  void *out = nullptr;
  bf::guard_mkl([&] {
    out = bf::new_event(
        blas::gemm(bf::Q(q), T(ta), T(tb), m, n, k, static_cast<TS>(alpha),
                   reinterpret_cast<const TA *>(a), lda, reinterpret_cast<const TA *>(b), ldb,
                   static_cast<TS>(beta), reinterpret_cast<TC *>(c), ldc,
                   bf::deps_chained(*bf::Store(q), deps, n_deps)),
        bf::Store(q));
  });
  return out;
}
} // namespace

#define BF_HGEMM(name, TA, TC, TS, CPTR)                                                        \
  BF_API void *name(void *q, int32_t ta, int32_t tb, int64_t m, int64_t n, int64_t k,          \
                    float alpha, const uint16_t *a, int64_t lda, const uint16_t *b,            \
                    int64_t ldb, float beta, CPTR *c, int64_t ldc, void *const *deps,          \
                    int64_t n_deps) {                                                          \
    return hgemm_async<TA, TC, TS>(q, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,    \
                                   deps, n_deps);                                              \
  }

BF_HGEMM(bf_hgemm_async, sycl::half, sycl::half, sycl::half, uint16_t)
BF_HGEMM(bf_hgemm_f32_async, sycl::half, float, float, float)
BF_HGEMM(bf_bgemm_async, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float, uint16_t)
BF_HGEMM(bf_bgemm_f32_async, oneapi::mkl::bfloat16, float, float, float)
