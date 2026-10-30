// Shared by the shim translation units: export macro, queue handle cast, error capture.
#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <cstdint>
#include <string>

#if defined(_WIN32)
#define BF_API extern "C" __declspec(dllexport)
#else
#define BF_API extern "C" __attribute__((visibility("default")))
#endif

namespace bf {
// One slot per thread; defined once in shim.cpp, read through bf_last_error().
extern thread_local std::string g_last_error;

inline sycl::queue &Q(void *q) { return *static_cast<sycl::queue *>(q); }

// Run f, mapping exceptions to status codes: 0 ok, 1 SYCL error, 2 oneMKL error, 3 other.
template <class F> int32_t guard(F &&f) {
  try {
    f();
    return 0;
  } catch (const sycl::exception &e) {
    g_last_error = e.what();
    return 1;
  } catch (const oneapi::mkl::exception &e) {
    g_last_error = e.what();
    return 2;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return 3;
  } catch (...) {
    g_last_error = "unknown C++ exception";
    return 3;
  }
}
} // namespace bf
