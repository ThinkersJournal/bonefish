// Standalone reproducer: N host threads doing oneMKL SYCL gemm. No Rust, no bonefish code.
// Used for docs/INTEL-REPORT-DRAFT.md; see that file for the results and the versions.
//
// Build (oneAPI DPC++ compiler, MKL on the include/library path, e.g. after setvars):
//   icpx -fsycl -O2 -qmkl=parallel repro.cpp -o repro
// Run each configuration in a FRESH process, many times: the failure is a race in the first
// concurrent use, so it only shows at process start.
//   repro <mode> <threads> <rounds> [alloc]
// mode: own     every thread creates its own queue, then loops (alloc + gemm + free)   <- fails
//       shared  one queue created on main, all threads use it                          <- fails
//       pre     one queue per thread, all created serially on main before threads     <- fails
//       warm    like pre, plus one serial gemm per queue on main before threads start <- passes
//       warm1   like warm but only the FIRST queue is warmed                          <- passes
//       locked  like own, with all gemm calls behind one std::mutex                   <- passes
// "alloc" as a 4th argument skips gemm and only allocates/frees (SYCL alone)          <- passes
// Exit codes: 0 ok, 3 wrong result, 42 watchdog (a hang), anything else is a crash.
// A watchdog prints each thread's last stage and exits 42 if the run exceeds 20 s.
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

static std::atomic<int> g_stage[64];
static const char *kStage[] = {"start", "queue-created", "alloc", "gemm-call", "gemm-done", "free", "finished"};

static bool g_gemm = true;
static bool g_locked = false;
static std::mutex g_mkl_mutex;
static void work(int t, sycl::queue *q, int rounds, bool own) {
  std::unique_ptr<sycl::queue> mine;
  if (own) {
    g_stage[t] = 0;
    mine = std::make_unique<sycl::queue>(sycl::cpu_selector_v);
    q = mine.get();
  }
  g_stage[t] = 1;
  const double scale = t + 2;
  for (int r = 0; r < rounds; r++) {
    g_stage[t] = 2;
    double *a = sycl::malloc_shared<double>(4, *q), *b = sycl::malloc_shared<double>(4, *q),
           *c = sycl::malloc_shared<double>(4, *q);
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
    b[0] = scale; b[1] = 0; b[2] = 0; b[3] = scale;
    c[0] = c[1] = c[2] = c[3] = 0;
    g_stage[t] = 3;
    std::unique_lock<std::mutex> lk(g_mkl_mutex, std::defer_lock);
    if (g_locked) lk.lock();
    if (g_gemm) oneapi::mkl::blas::column_major::gemm(*q, oneapi::mkl::transpose::nontrans,
                                          oneapi::mkl::transpose::nontrans, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2)
        .wait();
    if (lk.owns_lock()) lk.unlock();
    g_stage[t] = 4;
    if (g_gemm && (c[0] != scale || c[3] != 4 * scale)) { std::printf("WRONG RESULT thread %d\n", t); std::fflush(stdout); std::_Exit(3); }
    g_stage[t] = 5;
    sycl::free(a, *q); sycl::free(b, *q); sycl::free(c, *q);
  }
  g_stage[t] = 6;
}

int main(int argc, char **argv) {
  if (argc < 4) { std::printf("usage: repro own|shared|pre threads rounds\n"); return 2; }
  const std::string mode0 = argv[1];
  if (mode0 == "locked") g_locked = true;
  const std::string mode = (mode0 == "locked") ? std::string("own") : (mode0 == "warm1") ? std::string("warm1") : mode0;
  if (argc > 4 && std::string(argv[4]) == "alloc") g_gemm = false;
  const int threads = std::atoi(argv[2]), rounds = std::atoi(argv[3]);
  std::atomic<bool> done{false};
  std::thread dog([&] {
    for (int i = 0; i < 200 && !done; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!done) {
      std::printf("HANG mode=%s threads=%d. last stages:", mode.c_str(), threads);
      for (int t = 0; t < threads; t++) std::printf(" t%d=%s", t, kStage[g_stage[t].load()]);
      std::printf("\n");
      std::fflush(stdout);
      std::_Exit(42);
    }
  });
  std::vector<std::unique_ptr<sycl::queue>> pre;
  std::unique_ptr<sycl::queue> shared;
  if (mode == "shared") shared = std::make_unique<sycl::queue>(sycl::cpu_selector_v);
  if (mode == "pre" || mode == "warm" || mode == "warm1") for (int t = 0; t < threads; t++) pre.push_back(std::make_unique<sycl::queue>(sycl::cpu_selector_v));
  if (mode == "warm" || mode == "warm1") {  // serial gemm on main before any thread starts: all queues, or only the first
    for (auto &pq : pre) {
      if (mode == "warm1" && &pq != &pre[0]) break;
      double *a = sycl::malloc_shared<double>(4, *pq), *b = sycl::malloc_shared<double>(4, *pq), *c = sycl::malloc_shared<double>(4, *pq);
      for (int i = 0; i < 4; i++) a[i] = b[i] = c[i] = 1;
      oneapi::mkl::blas::column_major::gemm(*pq, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2).wait();
      sycl::free(a, *pq); sycl::free(b, *pq); sycl::free(c, *pq);
    }
  }
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; t++) {
    sycl::queue *q = mode == "shared" ? shared.get() : (mode == "pre" || mode == "warm" || mode == "warm1") ? pre[t].get() : nullptr;
    ts.emplace_back(work, t, q, rounds, mode == "own");
  }
  for (auto &th : ts) th.join();
  done = true;
  dog.join();
  std::printf("ok mode=%s threads=%d\n", mode.c_str(), threads);
  return 0;
}
