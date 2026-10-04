# Open review findings — round 15 (reviewer agent 94e52ea7, reviewed commit 1c0ad0b)

STATUS: both round-15 findings fixed; round-16 request sent. Fix
summary:

1. CRITICAL negative integer operands triggered lossy uint64
   conversion — FIXED. The uint64-preservation branch of
   `_vec_arith_dtype` now accepts ANY int literal (not only
   non-negative): modular uint64 arithmetic is exact whenever the true
   result is in [0, 2**64) — (2**63 + 3) + (-2) -> 2**63 + 1 exactly —
   and results outside that range are unrepresentable in any dtype.
   Signed ARRAYS still route through the promotion paths (per-lane
   mixed-sign results have no common dtype).
   Tests: test_uint64_negative_literal_arithmetic_exact.

2. CRITICAL bitwise expressions lost maybe-boolean tracking — FIXED.
   `_maybe_bool_result` handles bitwise BinOps (and/or/xor): bitwise
   expressions preserve boolean-ness (arithmetic results are already
   intified and numeric, so they are correctly excluded).
   `a = min(x, True); b = a & True; b + b` -> [2, 0].
   Tests: test_bitwise_maybe_bool_arithmetic,
   test_bitwise_maybe_bool_or, test_bitwise_maybe_bool_strict,
   test_arithmetic_maybe_bool_result_not_maybe_bool.

Round-15 regressions: tests/test_review_round15.py (5 tests). make
check green: 581 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged.
