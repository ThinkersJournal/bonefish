// Standalone reproducer, finding 2 of docs/INTEL-REPORT-DRAFT.md: on a queue created with
// sycl::property::queue::in_order, a oneMKL `dot` submitted right after another oneMKL call does
// not wait for it. No Rust, no bonefish code; one thread.
//
// Build (oneAPI DPC++ compiler, MKL on the include/library path, e.g. after setvars):
//   icpx -fsycl -O2 -qmkl=parallel repro_order.cpp -o repro_order
// Run (a fresh process is fine; the effect is not a first-use race):
//   repro_order <rounds>
//
// Each round allocates fresh shared USM, submits two calls back to back WITHOUT waiting between
// them, waits once, and checks whether the second call saw the first call's result. For every
// pair it runs two variants: A, no dependency between the calls; B, the second call lists the
// first call's event as an explicit dependency. Prints, per pair, how many rounds were wrong.
//
// Pairs (the second call reads what the first wrote):
//   axpy->dot   y = 2x + y, then dot(x, y)            right answer 70 (stale y gives 10)
//   copy->dot   y = x, then dot(y, y)                 right answer 30 (stale y gives 4)
//   gemm->dot   Z = 2I * ones, then dot(Z, Z)         right answer 64 (stale Z gives 0)
//   scal->axpy  x = 2x, then y = x + y                right answer [3,5,7,9] (control)
//   gemm->gemm  Z = 2I * ones, then W = 2I * Z        right answer all 4 (control)
// Exit code 0 always; read the numbers.
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace blas = oneapi::mkl::blas::column_major;
using sycl::event;

static double *mk(sycl::queue &q, const std::vector<double> &v) {
  double *p = sycl::malloc_shared<double>(v.size() ? v.size() : 1, q);
  for (size_t i = 0; i < v.size(); i++) p[i] = v[i];
  return p;
}

// Returns true when the second call saw the first call's result.
static bool round_ok(sycl::queue &q, const std::string &pair, bool with_dep) {
  double *x = mk(q, {1, 2, 3, 4});
  double *y = mk(q, {1, 1, 1, 1});
  double *r = mk(q, {0});
  std::vector<double> m(16, 0.0);
  for (int i = 0; i < 4; i++) m[i * 5] = 2.0;
  double *two_i = mk(q, m);
  double *ones = mk(q, std::vector<double>(16, 1.0));
  double *z = mk(q, std::vector<double>(16, 0.0));
  double *w = mk(q, std::vector<double>(16, 0.0));
  auto deps = [&](event e) { return with_dep ? std::vector<event>{e} : std::vector<event>{}; };
  bool ok = false;
  if (pair == "axpy->dot") {
    event e1 = blas::axpy(q, 4, 2.0, x, 1, y, 1);
    blas::dot(q, 4, x, 1, y, 1, r, deps(e1));
    q.wait();
    ok = r[0] == 70.0;
  } else if (pair == "copy->dot") {
    event e1 = blas::copy(q, 4, x, 1, y, 1);
    blas::dot(q, 4, y, 1, y, 1, r, deps(e1));
    q.wait();
    ok = r[0] == 30.0;
  } else if (pair == "gemm->dot") {
    event e1 = blas::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4,
                          4, 4, 1.0, two_i, 4, ones, 4, 0.0, z, 4);
    blas::dot(q, 16, z, 1, z, 1, r, deps(e1));
    q.wait();
    ok = r[0] == 64.0;
  } else if (pair == "scal->axpy") {
    event e1 = blas::scal(q, 4, 2.0, x, 1);
    blas::axpy(q, 4, 1.0, x, 1, y, 1, deps(e1));
    q.wait();
    ok = y[0] == 3.0 && y[1] == 5.0 && y[2] == 7.0 && y[3] == 9.0;
  } else {  // gemm->gemm
    event e1 = blas::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4,
                          4, 4, 1.0, two_i, 4, ones, 4, 0.0, z, 4);
    blas::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, 1.0,
               two_i, 4, z, 4, 0.0, w, 4, deps(e1));
    q.wait();
    ok = true;
    for (int i = 0; i < 16; i++) ok = ok && w[i] == 4.0;
  }
  for (double *p : {x, y, r, two_i, ones, z, w}) sycl::free(p, q);
  return ok;
}

int main(int argc, char **argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 100;
  sycl::queue q{sycl::cpu_selector_v, sycl::property_list{sycl::property::queue::in_order()}};
  std::printf("device: %s; in_order: %d; rounds per cell: %d\n",
              q.get_device().get_info<sycl::info::device::name>().c_str(), (int)q.is_in_order(),
              rounds);
  std::printf("%-12s %-22s %-22s\n", "pair", "wrong, no dependency", "wrong, explicit dep");
  for (const char *pair : {"axpy->dot", "copy->dot", "gemm->dot", "scal->axpy", "gemm->gemm"}) {
    int wrong_a = 0, wrong_b = 0;
    for (int i = 0; i < rounds; i++) {
      if (!round_ok(q, pair, false)) wrong_a++;
      if (!round_ok(q, pair, true)) wrong_b++;
    }
    std::printf("%-12s %-22d %-22d\n", pair, wrong_a, wrong_b);
  }
  return 0;
}
