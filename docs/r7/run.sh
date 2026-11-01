#!/usr/bin/env bash
# ROADMAP R7: build sweep.cpp once per floating-point flag set and print each table.
# Linux after `source /opt/intel/oneapi/setvars.sh`, or Windows in Git Bash with the oneAPI compiler
# and the MSVC/Windows SDK library paths on LIB. Select a device with ONEAPI_DEVICE_SELECTOR.
#   usage: run.sh
set -u
cd "$(dirname "$0")"
EXE=""
[ "${OS:-}" = "Windows_NT" ] && EXE=".exe"
icpx --version | head -1
sycl-ls
for variant in "default:" "precise:-ffp-model=precise" "no-fast-math:-fno-fast-math" "fp-model-precise:-fp-model=precise"; do
  name="${variant%%:*}"
  flags="${variant#*:}"
  echo "== flags: ${flags:-<none, compiler default>}"
  # shellcheck disable=SC2086
  icpx -fsycl -O2 $flags sweep.cpp -o "sweep_$name$EXE" || exit 1
  "./sweep_$name$EXE" || exit 1
done
