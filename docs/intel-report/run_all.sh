#!/usr/bin/env bash
# Builds the three reproducers and runs every measurement in docs/INTEL-REPORT-DRAFT.md, printing
# one table per finding. Linux, after `source /opt/intel/oneapi/setvars.sh`.
#   usage: run_all.sh [runs_per_cell]      (default 40; each run is a FRESH process)
# Exit codes of repro: 0 clean, 3 wrong result, 42 watchdog (hang), anything else is a crash.
set -u
cd "$(dirname "$0")"
RUNS="${1:-40}"

echo "== environment"
uname -srm
grep PRETTY_NAME /etc/os-release || true
lscpu | grep -E "Model name|^CPU\(s\)|Vendor"
icpx --version | head -1
grep -E "__INTEL_MKL__|__INTEL_MKL_MINOR__|__INTEL_MKL_UPDATE__" "${MKLROOT:-/opt/intel/oneapi/mkl/latest}/include/mkl_version.h" | head -3
sycl-ls

echo "== build (one command per file)"
icpx -fsycl -O2 -qmkl=parallel repro.cpp -o repro || exit 1
icpx -fsycl -O2 -qmkl=parallel repro_order.cpp -o repro_order || exit 1
icpx -fsycl -O2 -qmkl=parallel repro_dft.cpp -o repro_dft || exit 1

# cell <label> <command...>: run RUNS fresh processes, tally exit codes.
cell() {
  local label="$1"; shift
  local ok=0 wrong=0 hang=0 crash=0 codes=""
  for _ in $(seq 1 "$RUNS"); do
    timeout 60 "$@" >/dev/null 2>&1
    rc=$?
    case $rc in
      0) ok=$((ok + 1)) ;;
      3) wrong=$((wrong + 1)) ;;
      42 | 124) hang=$((hang + 1)) ;;
      *) crash=$((crash + 1)); codes="$codes $rc" ;;
    esac
  done
  printf '%-34s runs=%d clean=%d wrong=%d hang=%d crashed=%d%s\n' "$label" "$RUNS" "$ok" "$wrong" "$hang" "$crash" "${codes:+ (exit codes:$codes)}"
}

echo "== finding 1: gemm from 4 threads x 40 rounds, fresh process per run"
cell "own (cold)"        ./repro own 4 40
cell "pre"                ./repro pre 4 40
cell "shared"             ./repro shared 4 40
cell "own, 1 thread"      ./repro own 1 40
cell "own, 2 threads"     ./repro own 2 40
cell "own, alloc only"    ./repro own 4 40 alloc
cell "shared, alloc only" ./repro shared 4 40 alloc
cell "locked"             ./repro locked 4 40
cell "warm1"              ./repro warm1 4 40
cell "warm"               ./repro warm 4 40

echo "== finding 2: in-order queue ordering, 100 rounds per cell (wrong = stale read), 3 runs"
for i in 1 2 3; do echo "-- run $i"; ./repro_order 100; done

echo "== finding 3: zero-length DFT (a crash is exit 136 = SIGFPE)"
for prec in s d; do
  for inplace in 0 1; do
    ./repro_dft $prec $inplace 0 >/dev/null 2>&1; d=$?
    MKL_ENABLE_INSTRUCTIONS=AVX2 ./repro_dft $prec $inplace 0 >/dev/null 2>&1; a=$?
    echo "n=0 precision=$prec inplace=$inplace: default dispatch exit=$d, AVX2 exit=$a"
  done
done
