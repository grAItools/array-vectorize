# Open review findings — round 22 (reviewer agent 94e52ea7, reviewed commit b51db4c)

STATUS: the round-22 finding fixed; round-23 request sent. Fix summary:

1. CRITICAL positive signed-array divisors caused lossy float64
   remainder and floor division with huge uint64 dividends — FIXED.
   The non-negative signed path in `_vec_arith` now covers ALL five
   integer operators (add/sub/mul/floordiv/mod): non-negative signed
   values are faithful in uint64 — add/sub/mul are modular-exact, and
   %,// match Python semantics for non-negative dividends and
   divisors.
   Repros: max(uint64[2**63+3], True) % int64[2] -> 1 and
   max(uint64[2**63+3], True) // int64[2] -> 4611686018427387905,
   exactly, on numpy AND array_api_strict (previously the rounded
   float64 values [0.] and 4611686018427387904). Non-negative signed
   LEFT operands also benefit: int64[7] % max(uint64[2**63+1], True)
   -> 7. Negative signed values keep the layered round-21 behavior
   (int64 when the uint64 values fit, documented float64 fallback
   otherwise).
   Tests: test_uint64_mod_positive_int64_array_exact,
   test_uint64_floordiv_positive_int64_array_exact,
   test_uint64_mod_positive_int64_array_strict,
   test_uint64_floordiv_positive_int64_array_strict,
   test_nonnegative_signed_left_mod_uint64_exact.

Round-22 regressions: tests/test_review_round22.py (5 tests). make
check green: 621 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged
(helper-internal logic only).
