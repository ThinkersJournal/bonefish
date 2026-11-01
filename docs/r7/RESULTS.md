# R7: float unary math, default floating-point model vs `-ffp-model=precise`

Harness: `sweep.cpp` (plain C++ with SYCL), driver `run.sh`; Linux numbers come from the manual
workflow `.github/workflows/r7-sweep.yml`. For each op, 8192 float inputs evenly spaced over the
domain given in the `OPS` table of the source run on the device in two spellings (the libm
name, e.g. `expf`, and `sycl::exp`); each result is compared with the same function in double
precision on the host. Error is in ULPs of the correctly rounded float (0.5 is the floor). Max over
the 8192 inputs.

Measured on CPU (OpenCL) devices only:

| | OS | CPU | compiler |
|---|---|---|---|
| Windows | Windows 11 | AMD Ryzen 9 7940HX | icpx 2026.1.0.20260617, OpenCL RT 2026.21.7.0.24 |
| Linux | Ubuntu (GitHub-hosted runner) | AMD EPYC 9V74 | icpx 2026.2.0.20261001 |

**Both machines gave the identical table for all four flag sets (every cell equal).**

Flag sets: compiler default (no flags), `-ffp-model=precise`, `-fno-fast-math`, `-fp-model=precise`.
The last three gave identical tables to each other (every op, both spellings).

Max ULP error (compiler default = default flags; precise = any of the three precise spellings):

| op | libm, default | sycl::, default | libm, precise | sycl::, precise |
|---|---|---|---|---|
| exp | 7620.53 | 1.28 | 1.28 | 1.28 |
| exp2 | 28.75 | 1.13 | 1.09 | 1.09 |
| expm1 | 1.00 | 1.00 | 1.02 | 1.02 |
| log | 20.77 | 1.07 | 1.07 | 1.07 |
| log2 | 16.50 | 1.00 | 1.00 | 1.00 |
| log10 | 19.97 | 1.00 | 1.00 | 1.00 |
| log1p | 1.00 | 1.00 | 1.00 | 1.00 |
| sqrt | 1.00 | 1.00 | 1.00 | 1.00 |
| cbrt | 1.29 | 1.29 | 1.29 | 1.29 |
| sin | 5046.50 | 4.31 | 3.78 | 3.78 |
| sin_wide | 5046.80 | 3.89 | 4.24 | 4.24 |
| cos | 5070.21 | 4.18 | 3.60 | 3.60 |
| cos_wide | 5069.82 | 3.99 | 3.99 | 3.99 |
| tan | 1579.09 | 2.79 | 2.79 | 2.79 |
| asin | 4.34 | 4.34 | 4.34 | 4.34 |
| acos | 2.27 | 2.27 | 2.27 | 2.27 |
| atan | 1.94 | 1.94 | 1.94 | 1.94 |
| sinh | 2.70 | 2.70 | 2.70 | 2.70 |
| cosh | 3.78 | 3.78 | 3.54 | 3.54 |
| tanh | 2.62 | 2.62 | 2.62 | 2.62 |
| asinh | 1.01 | 1.01 | 1.01 | 1.01 |
| acosh | 1.29 | 1.29 | 1.29 | 1.29 |
| atanh | 1.77 | 1.77 | 1.77 | 1.77 |
| erf | 7.94 | 7.94 | 7.94 | 7.94 |
| erfc | 2.75 | 2.75 | 2.75 | 2.75 |
| tgamma | 1.00 | 1.00 | 1.00 | 1.00 |
| lgamma | 1.00 | 1.00 | 1.01 | 1.01 |

Reading it:
- Under default flags the **libm spelling is badly wrong for 8 functions (10 of the 27 rows, `sin` and `cos` are also run over a wide range)**: `exp` (7620 ULP), `sin`/`cos` (about 5000), `tan` (1579), `exp2` (29), `log` (21), `log10` (20), `log2` (16.5). The same ops are within 1.0 to 4.3 ULP with any precise flag set, or with the `sycl::` spelling under default flags.
- With `sycl::` spelling, default flags are fine (max 7.94, from `erf`), though a few ops move by a fraction of a ULP between default and precise (`sin` 4.31 vs 3.78, `cos` 4.18 vs 3.60, `sin_wide` 3.89 vs 4.24).
- `erf` is 7.94 ULP in every cell, so it is not a fast-math effect: `sycl::erf` itself is not correctly rounded here.
- The other 17 rows differ between libm and `sycl::` spelling, or between default and precise, by at most 0.25 ULP (largest: `cosh`, 3.78 vs 3.54).

Not measured: double precision, binary ops (`pow`, `atan2`), vector types, any GPU (Intel GPU not available
to this project; GPU math lowering may differ), other oneAPI releases, macOS.
