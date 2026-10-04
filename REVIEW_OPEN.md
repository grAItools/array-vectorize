# Open review findings — round 20 (reviewer agent 94e52ea7, reviewed commit 68805f4)

STATUS: both round-20 findings fixed; round-21 request sent. Fix
summary:

1. MAJOR remainder by int64-min raised despite representable inputs —
   FIXED. The negative-divisor remainder now ADDS the negative divisor
   directly (`(a %% |d|) + d`) instead of subtracting the negated
   positive (which forms |d| = 2**63, unrepresentable as an int64
   scalar): max(x, True) % (-(2**63)) -> -9223372036854775807 on numpy
   AND array_api_strict.
   Tests: test_uint64_mod_int64_min_divisor,
   test_uint64_mod_int64_min_divisor_strict.

2. MAJOR arithmetic helper results lost kind information — FIXED.
   `_numeric_kind` infers `_vec_arith` result kinds: true division and
   float operands give 'float'; known int/bool operands give 'int'
   (bools are converted inside the helper, so results are never bool);
   unknown operands stay None. Provably-integer arithmetic results
   therefore keep receiving the negative-exponent float cast:
   `a = (x > 0) + 1; a ** -1` -> 0.5 on both backends.
   Tests: test_arith_result_negative_power,
   test_arith_result_negative_power_strict,
   test_arith_result_float_kind.

Round-20 regressions: tests/test_review_round20.py (5 tests). make
check green: 611 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged.
