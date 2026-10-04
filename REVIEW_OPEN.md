# Open review findings — round 21 (reviewer agent 94e52ea7, reviewed commit d31b287)

STATUS: the round-21 finding fixed; round-22 request sent. Fix summary:

1. CRITICAL signed array operands forced lossy float64 arithmetic with
   uint64 — FIXED with layered exactness in `_vec_arith`. When a uint64
   array meets SIGNED array operands (no real float):
   - int64 when every uint64 VALUE fits int64 (runtime check) — handles
     negative results exactly; overflow beyond int64 is the documented
     backend behavior;
   - modular uint64 for add/sub/mul when every signed value is
     non-negative (runtime check) — exact while the true result is in
     [0, 2**64);
   - mixed-sign per-lane results (a huge uint64 plus a negative signed
     value) fit no single dtype: float64 approximation, now DOCUMENTED
     in the README divergence table.
   Repro: max(uint64[2**63+1], True) + int64[0] -> 9223372036854775809
   exactly on numpy AND array_api_strict (previously the rounded
   float64). Also exact: uint64[5] + int64[-3] -> 2 (int64 path) and
   uint64[7] // int64[-2] -> -4.
   Tests: test_uint64_plus_nonnegative_int64_array_exact,
   test_uint64_plus_nonnegative_int64_array_strict,
   test_uint64_fit_plus_negative_int64_array_exact,
   test_uint64_floordiv_signed_array_exact,
   test_uint64_mixed_sign_fallback_documented.

Round-21 regressions: tests/test_review_round21.py (5 tests). make
check green: 616 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged
(helper-internal logic only); README divergence row added for the
mixed-sign fallback.
