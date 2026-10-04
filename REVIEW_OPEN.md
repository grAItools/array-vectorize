# Open review findings — round 17 (reviewer agent 94e52ea7, reviewed commit 2aa485d)

STATUS: both round-17 findings fixed; round-18 request sent. Fix
summary:

1. CRITICAL float64 fallback corrupted integer remainder/floor-division/
   negation for large uint64 — FIXED with runtime value checks. When a
   uint64 array meets integer literals in floordiv/mod with a negative
   divisor (or negation), `_vec_arith_dtype` now returns INT64 whenever
   every uint64 VALUE fits int64 (checked at runtime via
   xp.all(a < 2**63)) — exact for those, and their results fit int64:
   (2**53+1) % -2 -> -1; (2**53+1) // -2 -> -4503599627370497;
   -(2**53+1) -> -9007199254740993. Values beyond int64 fall back to
   float64 (documented best effort: those results are unrepresentable).
   Tests: test_uint64_mod_negative_large_exact,
   test_uint64_floordiv_negative_large_exact,
   test_uint64_negate_large_exact,
   test_uint64_negate_huge_best_effort_float.

2. MAJOR true division selected integer operands for boolean/integer
   inputs — FIXED. `_vec_arith_dtype` returns a FLOAT dtype for true
   division in every path (float64 when any int-class operand is
   present, else the widest float): min(x, True) / 2 -> [0.5, 0.0] on
   array_api_strict (previously TypeError) and numpy.
   Tests: test_truediv_bool_strict, test_truediv_bool_numpy.

Round-17 regressions: tests/test_review_round17.py (6 tests). make
check green: 593 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged.
