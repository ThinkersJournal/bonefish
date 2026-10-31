// Real-to-complex and complex-to-real DFT over oneapi::mkl::dft (R3b). Out-of-place only: the
// forward transform reads `prod(lengths)` reals and writes the conjugate-even half spectrum, whose
// LAST dimension has `lengths[d-1] / 2 + 1` complex entries (the rest of the layout is the plain
// row-major one, last dimension contiguous); the backward transform goes the other way. These are
// oneMKL's defaults for an out-of-place real descriptor, which the tests check against a naive
// DFT instead of taking on trust; whether a GPU lays the data out the same way is UNVERIFIED.
// In-place real transforms need padded storage and are not offered.
//
// The handle owns a committed descriptor and a copy of the queue, like the complex one.
#include "common.hpp"

#include <complex>
#include <memory>
#include <vector>

namespace dft = oneapi::mkl::dft;

namespace {
using DescRF = dft::descriptor<dft::precision::SINGLE, dft::domain::REAL>;
using DescRD = dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL>;

struct DftR {
  sycl::queue q;
  // The creating queue's error store, so an event from `bf_dftr_compute_async` can report that
  // queue's asynchronous errors.
  std::shared_ptr<bf::ErrStore> store;
  bool dbl;
  std::unique_ptr<DescRF> f;
  std::unique_ptr<DescRD> d;
};

template <class Desc, class T>
std::unique_ptr<Desc> make(sycl::queue &q, int64_t ndim, const int64_t *lengths, double fwd_scale,
                           double bwd_scale) {
  auto desc = std::make_unique<Desc>(std::vector<std::int64_t>(lengths, lengths + ndim));
  desc->set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);
  // Dense row-major strides, set explicitly. Measured (Windows, oneAPI 2026.1, CPU device): the
  // DEFAULT real-domain input layout pads each row to 2 * (n_last / 2 + 1) reals even out of place,
  // so a dense real array (what a caller naturally has) was silently read with the wrong row
  // stride: a delta at element (1, 0) of a 4 x 6 array was treated as padding and ignored. A
  // stride vector is {offset, stride of dim 0, ..., stride of the last dim}. FWD_STRIDES describes
  // the real (forward-domain) data, BWD_STRIDES the half-spectrum complex data, whose last
  // dimension has n_last / 2 + 1 entries.
  std::vector<std::int64_t> real_strides(static_cast<size_t>(ndim) + 1, 0);
  std::vector<std::int64_t> cplx_strides(static_cast<size_t>(ndim) + 1, 0);
  real_strides[static_cast<size_t>(ndim)] = 1;
  cplx_strides[static_cast<size_t>(ndim)] = 1;
  for (int64_t j = ndim - 1; j >= 1; j--) {
    const int64_t real_extent = lengths[j];
    const int64_t cplx_extent = (j == ndim - 1) ? lengths[j] / 2 + 1 : lengths[j];
    real_strides[static_cast<size_t>(j)] = real_strides[static_cast<size_t>(j) + 1] * real_extent;
    cplx_strides[static_cast<size_t>(j)] = cplx_strides[static_cast<size_t>(j) + 1] * cplx_extent;
  }
  desc->set_value(dft::config_param::FWD_STRIDES, real_strides);
  desc->set_value(dft::config_param::BWD_STRIDES, cplx_strides);
  desc->set_value(dft::config_param::FORWARD_SCALE, static_cast<T>(fwd_scale));
  desc->set_value(dft::config_param::BACKWARD_SCALE, static_cast<T>(bwd_scale));
  desc->commit(q);
  return desc;
}

// dir 0: forward, `in` is T[prod(lengths)], `out` is complex<T>[half spectrum].
// dir 1: backward, `in` is complex<T>[half spectrum], `out` is T[prod(lengths)].
template <class Desc, class T>
sycl::event run(Desc &desc, bool forward, void *in, void *out, const std::vector<sycl::event> &deps) {
  if (forward) {
    return dft::compute_forward(desc, static_cast<T *>(in), static_cast<std::complex<T> *>(out),
                                deps);
  }
  return dft::compute_backward(desc, static_cast<std::complex<T> *>(in), static_cast<T *>(out),
                               deps);
}

sycl::event run_any(DftR &p, bool forward, void *in, void *out,
                    const std::vector<sycl::event> &deps) {
  return p.dbl ? run<DescRD, double>(*p.d, forward, in, out, deps)
               : run<DescRF, float>(*p.f, forward, in, out, deps);
}
} // namespace

// Committed real DFT descriptor over `ndim` `lengths`. prec: 0 single, 1 double. The scales
// multiply the forward / backward result (oneMKL's defaults are 1 and 1). Null on failure; a
// length below 1 is refused here, before oneMKL (see bf::check_dft_lengths).
BF_API void *bf_dftr_new(void *q, int32_t prec, int64_t ndim, const int64_t *lengths,
                         double fwd_scale, double bwd_scale) {
  DftR *h = nullptr;
  bf::guard_mkl([&] {
    bf::check_dft_lengths(ndim, lengths);
    auto p = std::make_unique<DftR>(DftR{bf::Q(q), bf::Store(q), prec == 1, nullptr, nullptr});
    if (prec == 1) {
      p->d = make<DescRD, double>(p->q, ndim, lengths, fwd_scale, bwd_scale);
    } else {
      p->f = make<DescRF, float>(p->q, ndim, lengths, fwd_scale, bwd_scale);
    }
    h = p.release();
  });
  return h;
}

BF_API void bf_dftr_free(void *h) {
  // The descriptor destructor calls into oneMKL, so it takes the same lock.
  bf::MklLock lock;
  delete static_cast<DftR *>(h);
}

// Blocking transform on USM data. dir: 0 forward (real in, half-spectrum complex out), 1 backward
// (half-spectrum complex in, real out).
BF_API int32_t bf_dftr_compute(void *h, int32_t dir, void *in, void *out) {
  return bf::guard_mkl([&] { run_any(*static_cast<DftR *>(h), dir == 0, in, out, {}).wait(); });
}

// Non-blocking: as bf_dftr_compute, but returns the event (null on failure, see bf_last_error) and
// waits for `deps` (`n_deps` event handles, null with 0 for none). Only the submission runs under
// the oneMKL lock; the caller keeps the descriptor and the data alive, and the data untouched by the
// host, until the event or the queue has been waited on.
BF_API void *bf_dftr_compute_async(void *h, int32_t dir, void *in, void *out, void *const *deps,
                                   int64_t n_deps) {
  void *ev = nullptr;
  bf::guard_mkl([&] {
    auto *p = static_cast<DftR *>(h);
    auto d = bf::deps_chained(*p->store, deps, n_deps);
    ev = bf::new_event(run_any(*p, dir == 0, in, out, d), p->store);
  });
  return ev;
}
