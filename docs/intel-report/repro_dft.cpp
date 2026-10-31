// Standalone reproducer, finding 3 of docs/INTEL-REPORT-DRAFT.md: oneMKL SYCL DFT (complex to
// complex, CPU device) crashes with SIGFPE on some Intel Xeon CPUs with default instruction-set
// dispatch. Plain C++ with SYCL and oneMKL only; one thread; one configuration per process (a crash kills
// the process, so the caller loops over configurations).
//
// Build (oneAPI DPC++ compiler, MKL on the include/library path, e.g. after setvars):
//   icpx -fsycl -O2 -qmkl=parallel repro_dft.cpp -o repro_dft
// Run:
//   repro_dft <s|d> <inplace 0|1> <n> [rounds]
// Computes the forward DFT of a delta at index 1 of length n (X[k] = exp(-2 pi i k / n)) and checks
// it. Prints "ok" or "WRONG max_err=..." or "threw: ..." (an invalid length is refused with an
// exception) and exits 0 / 3 / 4. A crash (SIGFPE = exit 136 on Linux) is
// the finding. Try also MKL_ENABLE_INSTRUCTIONS=AVX2 (or SSE4_2, AVX512) in the environment.
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace dft = oneapi::mkl::dft;

template <class T> static int run(bool inplace, std::int64_t n, int rounds) {
  using Desc = dft::descriptor<std::is_same<T, float>::value ? dft::precision::SINGLE
                                                             : dft::precision::DOUBLE,
                               dft::domain::COMPLEX>;
  sycl::queue q{sycl::cpu_selector_v};
  double worst = 0.0;
  for (int r = 0; r < rounds; r++) {
    // Stage markers, so a crash can be placed: the last one printed is the last one reached.
    std::puts("stage: constructing the descriptor");
    std::fflush(stdout);
    Desc desc(n);
    std::puts("stage: descriptor constructed; setting placement");
    std::fflush(stdout);
    desc.set_value(dft::config_param::PLACEMENT,
                   inplace ? dft::config_value::INPLACE : dft::config_value::NOT_INPLACE);
    std::puts("stage: committing");
    std::fflush(stdout);
    desc.commit(q);
    std::puts("stage: committed");
    std::fflush(stdout);
    auto *in = sycl::malloc_shared<std::complex<T>>(n, q);
    auto *out = inplace ? in : sycl::malloc_shared<std::complex<T>>(n, q);
    for (std::int64_t i = 0; i < n; i++) in[i] = {0, 0};
    in[1 % n] = {1, 0};
    if (inplace) dft::compute_forward(desc, in).wait();
    else dft::compute_forward(desc, in, out).wait();
    const double pi = std::acos(-1.0);
    for (std::int64_t k = 0; k < n; k++) {
      std::complex<double> want = std::polar(1.0, -2.0 * pi * double(k) * double(1 % n) / double(n));
      worst = std::fmax(worst, std::abs(std::complex<double>(out[k]) - want));
    }
    sycl::free(in, q);
    if (!inplace) sycl::free(out, q);
  }
  const double tol = std::is_same<T, float>::value ? 1e-4 : 1e-10;
  if (worst > tol) {
    std::printf("WRONG max_err=%g\n", worst);
    return 3;
  }
  std::printf("ok\n");
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::printf("usage: repro_dft <s|d> <inplace 0|1> <n> [rounds]\n");
    return 2;
  }
  const bool single = std::string(argv[1]) == "s";
  const bool inplace = std::atoi(argv[2]) != 0;
  const std::int64_t n = std::atoll(argv[3]);
  const int rounds = argc > 4 ? std::atoi(argv[4]) : 1;
  // An invalid length (0 or negative) is expected to be reported as an exception, which is the
  // documented way for oneMKL to refuse a bad descriptor. "threw" is therefore a fine outcome;
  // a crash (SIGFPE, exit 136) is the finding.
  try {
    return single ? run<float>(inplace, n, rounds) : run<double>(inplace, n, rounds);
  } catch (const std::exception &e) {
    std::printf("threw: %s\n", e.what());
    return 4;
  }
}
