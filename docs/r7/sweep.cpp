// ROADMAP R7: accuracy of float unary math under the compiler's default floating-point model versus
// -ffp-model=precise, for two spellings of each function: the libm name (expf) and sycl::exp.
// Plain C++ with SYCL only. Build once per flag set (see run.sh) and compare the tables.
//
// For each op, N float inputs spread evenly over a domain are run on the device; each result is
// compared with the same function evaluated in double precision on the host. The error is reported
// in ULPs of the correctly rounded float of that double value (0.5 is the best any function can do;
// correctly rounded results show <= 0.5). Printed: max ULP error per spelling.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int N = 8192;

// Error in ULPs of the float nearest to `ref`.
static double ulp_err(float got, double ref) {
  float rf = static_cast<float>(ref);
  if (std::isnan(ref)) return std::isnan(got) ? 0.0 : 1e30;
  if (std::isinf(ref)) return (got == rf) ? 0.0 : 1e30;
  if (std::isnan(got) || std::isinf(got)) return 1e30;
  float up = std::nextafter(rf, INFINITY);
  float dn = std::nextafter(rf, -INFINITY);
  double ulp = std::fmax(static_cast<double>(up) - rf, static_cast<double>(rf) - dn) / 2.0;
  if (ulp == 0.0) ulp = std::ldexp(1.0, -149);
  return std::fabs(static_cast<double>(got) - ref) / ulp;
}

struct Row {
  std::string name;
  double libm;
  double syc;
};

#define OPS(X)                                                                                    \
  X(exp, expf(x), sycl::exp(x), std::exp(d), -10.0f, 10.0f)                                       \
  X(exp2, exp2f(x), sycl::exp2(x), std::exp2(d), -20.0f, 20.0f)                                   \
  X(expm1, expm1f(x), sycl::expm1(x), std::expm1(d), -5.0f, 5.0f)                                 \
  X(log, logf(x), sycl::log(x), std::log(d), 1e-3f, 1e3f)                                         \
  X(log2, log2f(x), sycl::log2(x), std::log2(d), 1e-3f, 1e3f)                                     \
  X(log10, log10f(x), sycl::log10(x), std::log10(d), 1e-3f, 1e3f)                                 \
  X(log1p, log1pf(x), sycl::log1p(x), std::log1p(d), -0.9f, 10.0f)                                \
  X(sqrt, sqrtf(x), sycl::sqrt(x), std::sqrt(d), 0.0f, 1e4f)                                      \
  X(cbrt, cbrtf(x), sycl::cbrt(x), std::cbrt(d), -1e3f, 1e3f)                                     \
  X(sin, sinf(x), sycl::sin(x), std::sin(d), -6.2831853f, 6.2831853f)                             \
  X(sin_wide, sinf(x), sycl::sin(x), std::sin(d), -100.0f, 100.0f)                                \
  X(cos, cosf(x), sycl::cos(x), std::cos(d), -6.2831853f, 6.2831853f)                             \
  X(cos_wide, cosf(x), sycl::cos(x), std::cos(d), -100.0f, 100.0f)                                \
  X(tan, tanf(x), sycl::tan(x), std::tan(d), -1.4f, 1.4f)                                         \
  X(asin, asinf(x), sycl::asin(x), std::asin(d), -1.0f, 1.0f)                                     \
  X(acos, acosf(x), sycl::acos(x), std::acos(d), -1.0f, 1.0f)                                     \
  X(atan, atanf(x), sycl::atan(x), std::atan(d), -100.0f, 100.0f)                                 \
  X(sinh, sinhf(x), sycl::sinh(x), std::sinh(d), -10.0f, 10.0f)                                   \
  X(cosh, coshf(x), sycl::cosh(x), std::cosh(d), -10.0f, 10.0f)                                   \
  X(tanh, tanhf(x), sycl::tanh(x), std::tanh(d), -10.0f, 10.0f)                                   \
  X(asinh, asinhf(x), sycl::asinh(x), std::asinh(d), -100.0f, 100.0f)                             \
  X(acosh, acoshf(x), sycl::acosh(x), std::acosh(d), 1.0f, 100.0f)                                \
  X(atanh, atanhf(x), sycl::atanh(x), std::atanh(d), -0.99f, 0.99f)                               \
  X(erf, erff(x), sycl::erf(x), std::erf(d), -4.0f, 4.0f)                                         \
  X(erfc, erfcf(x), sycl::erfc(x), std::erfc(d), -4.0f, 8.0f)                                     \
  X(tgamma, tgammaf(x), sycl::tgamma(x), std::tgamma(d), 0.1f, 30.0f)                             \
  X(lgamma, lgammaf(x), sycl::lgamma(x), std::lgamma(d), 0.1f, 100.0f)

int main() {
  sycl::queue q{sycl::default_selector_v};
  {
    sycl::device d = q.get_device();
    const char *ty = d.is_cpu() ? "cpu" : d.is_gpu() ? "gpu" : "other";
    std::printf("device: %s (%s, backend %d)\n", d.get_info<sycl::info::device::name>().c_str(), ty,
                static_cast<int>(d.get_backend()));
  }
  std::printf("%-10s %14s %14s\n", "op", "libm-spelling", "sycl::-spelling");
  float *in = sycl::malloc_shared<float>(N, q);
  float *o1 = sycl::malloc_shared<float>(N, q);
  float *o2 = sycl::malloc_shared<float>(N, q);
  std::vector<Row> rows;
#define RUN(NAME, LIBM, SYCLF, REF, LO, HI)                                                       \
  {                                                                                               \
    for (int i = 0; i < N; i++) in[i] = (LO) + ((HI) - (LO)) * (static_cast<float>(i) / (N - 1)); \
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {                                       \
       float x = in[id];                                                                          \
       o1[id] = LIBM;                                                                             \
       o2[id] = SYCLF;                                                                            \
     }).wait();                                                                                   \
    double m1 = 0, m2 = 0;                                                                        \
    for (int i = 0; i < N; i++) {                                                                 \
      double d = static_cast<double>(in[i]);                                                      \
      double r = REF;                                                                             \
      m1 = std::fmax(m1, ulp_err(o1[i], r));                                                      \
      m2 = std::fmax(m2, ulp_err(o2[i], r));                                                      \
    }                                                                                             \
    std::printf("%-10s %14.2f %14.2f\n", #NAME, m1, m2);                                          \
  }
  OPS(RUN)
  sycl::free(in, q);
  sycl::free(o1, q);
  sycl::free(o2, q);
  return 0;
}
