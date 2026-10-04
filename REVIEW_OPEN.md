# Open review findings — round 23 (reviewer agent 94e52ea7, reviewed commit 8d25c2e)

STATUS: the round-23 finding fixed; round-24 request sent. Fix summary:

1. CRITICAL non-negative operands did not guarantee an unsigned
   subtraction result — FIXED with result-aware subtraction. In the
   non-negative signed-array path, sub now:
   - returns uint64 when every per-lane difference is non-negative
     (runtime check), preserving u64-only positive results;
   - returns int64 when every per-lane difference fits int64
     (positive side <= int64-max, negative magnitude <= 2**63,
     checked at runtime): int64 subtraction is mod-2**64, so the
     wrapped uint64 casts still yield the EXACT value — including the
     int64-min edge (0 - 2**63 -> -(2**63));
   - falls through to the documented float64 only for genuinely
     mixed-magnitude batches (a u64-only positive in one lane and an
     int64-only negative in another), which fit no single dtype.
   Repro: int64[2**63-1] - max(uint64[2**63], True) -> -1 exactly on
   numpy AND array_api_strict (previously the wrapped 2**64-1). Also
   exact now: uint64[5] - int64[7] -> -2 (previously also silently
   wrapped). add/mul/floordiv/mod keep the round-21/22 behavior.
   Tests: test_int64_minus_uint64_huge_exact,
   test_int64_minus_uint64_huge_strict, test_sub_result_int64_min_exact,
   test_uint64_minus_signed_negative_result_exact,
   test_sub_nonnegative_result_keeps_uint64,
   test_sub_mixed_magnitude_fallback_documented.

Round-23 regressions: tests/test_review_round23.py (6 tests). make
check green: 627 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged
(helper-internal logic only).
