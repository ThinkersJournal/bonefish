// Complex-to-complex DFT over oneapi::mkl::dft. The handle owns a committed descriptor and a copy
// of the queue (a sycl::queue is a reference-counted handle, so the caller may free its own).
#include "common.hpp"

#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>

namespace dft = oneapi::mkl::dft;

namespace {
using DescF = dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX>;
using DescD = dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX>;

struct Dft {
  sycl::queue q;
  // The creating queue's error store, so an event from `bf_dft_compute_async` can report that
  // queue's asynchronous errors (the sycl::queue above is a plain copy and does not carry it).
  std::shared_ptr<bf::ErrStore> store;
  bool dbl;
  std::unique_ptr<DescF> f;
  std::unique_ptr<DescD> d;
};

template <class Desc, class T>
sycl::event run(Desc &desc, bool forward, void *in, void *out) {
  auto *i = static_cast<std::complex<T> *>(in);
  auto *o = static_cast<std::complex<T> *>(out);
  if (o == nullptr) return forward ? dft::compute_forward(desc, i) : dft::compute_backward(desc, i);
  return forward ? dft::compute_forward(desc, i, o) : dft::compute_backward(desc, i, o);
}

template <class Desc, class T>
sycl::event run_with(Desc &desc, bool forward, void *in, void *out,
                     const std::vector<sycl::event> &deps) {
  auto *i = static_cast<std::complex<T> *>(in);
  auto *o = static_cast<std::complex<T> *>(out);
  if (o == nullptr)
    return forward ? dft::compute_forward(desc, i, deps) : dft::compute_backward(desc, i, deps);
  return forward ? dft::compute_forward(desc, i, o, deps) : dft::compute_backward(desc, i, o, deps);
}

template <class Desc, class T>
std::unique_ptr<Desc> make(sycl::queue &q, int64_t ndim, const int64_t *lengths, bool inplace,
                           double fwd_scale, double bwd_scale) {
  auto desc = std::make_unique<Desc>(std::vector<std::int64_t>(lengths, lengths + ndim));
  desc->set_value(dft::config_param::PLACEMENT, inplace ? dft::config_value::INPLACE
                                                        : dft::config_value::NOT_INPLACE);
  desc->set_value(dft::config_param::FORWARD_SCALE, static_cast<T>(fwd_scale));
  desc->set_value(dft::config_param::BACKWARD_SCALE, static_cast<T>(bwd_scale));
  desc->commit(q);
  return desc;
}
} // namespace

// Committed complex-to-complex descriptor. prec: 0 single, 1 double. `lengths` has `ndim` entries.
// Scales multiply the forward / backward result (oneMKL's defaults are 1 and 1). Null on failure.
BF_API void *bf_dft_c2c_new(void *q, int32_t prec, int64_t ndim, const int64_t *lengths,
                            int32_t inplace, double fwd_scale, double bwd_scale) {
  Dft *h = nullptr;
  bf::guard_mkl([&] {
    if (ndim < 1 || lengths == nullptr) throw std::invalid_argument("dft: need at least one length");
    auto p = std::make_unique<Dft>(Dft{bf::Q(q), bf::Store(q), prec == 1, nullptr, nullptr});
    if (prec == 1) {
      p->d = make<DescD, double>(p->q, ndim, lengths, inplace != 0, fwd_scale, bwd_scale);
    } else {
      p->f = make<DescF, float>(p->q, ndim, lengths, inplace != 0, fwd_scale, bwd_scale);
    }
    h = p.release();
  });
  return h;
}

BF_API void bf_dft_free(void *h) {
  // The descriptor destructor calls into oneMKL, so it takes the same lock.
  bf::MklLock lock;
  delete static_cast<Dft *>(h);
}

// Blocking transform on USM interleaved complex data. For an in-place descriptor pass out = null
// (in is overwritten); for an out-of-place one pass both. dir: 0 forward, 1 backward.
BF_API int32_t bf_dft_compute(void *h, int32_t dir, void *in, void *out) {
  return bf::guard_mkl([&] {
    auto *p = static_cast<Dft *>(h);
    bool fwd = dir == 0;
    (p->dbl ? run<DescD, double>(*p->d, fwd, in, out) : run<DescF, float>(*p->f, fwd, in, out))
        .wait();
  });
}

// Non-blocking transform: as bf_dft_compute, but returns the event (null on failure, see
// bf_last_error) and waits for `deps` (`n_deps` event handles, null with 0 for none). Only the
// submission runs under the oneMKL lock. The caller keeps the descriptor, `in` and `out` alive,
// and the data untouched by the host, until the event or the queue has been waited on.
BF_API void *bf_dft_compute_async(void *h, int32_t dir, void *in, void *out, void *const *deps,
                                  int64_t n_deps) {
  void *ev = nullptr;
  bf::guard_mkl([&] {
    auto *p = static_cast<Dft *>(h);
    bool fwd = dir == 0;
    auto d = bf::deps_chained(*p->store, deps, n_deps);
    sycl::event e = p->dbl ? run_with<DescD, double>(*p->d, fwd, in, out, d)
                           : run_with<DescF, float>(*p->f, fwd, in, out, d);
    ev = bf::new_event(e, p->store);
  });
  return ev;
}
