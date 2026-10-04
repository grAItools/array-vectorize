# Open review findings — round 24 (reviewer agent 94e52ea7, reviewed commit 405770a)

STATUS: the round-24 finding fixed; round-25 request sent. Fix summary:

1. CRITICAL result-aware subtraction was missing from the
   literal/unsigned-array path — FIXED. The result-aware logic is
   extracted into `_u64_sub_exact` and used by EVERY integer
   subtraction path in `_vec_arith`: the literal/unsigned-array path
   (others_intish), the non-negative signed-array path, and (via the
   int64-fit branch) the signed path. uint64 when every per-lane
   difference is non-negative; exact mod-2**64 int64 when every
   per-lane difference fits int64 (including the int64-min edge);
   the documented float64 only for mixed-magnitude batches that fit
   no single dtype.
   Repros: 0 - max(uint64[1], True) -> -1 and
   max(uint64[5], True) - uint64[7] -> -2, exactly, on numpy AND
   array_api_strict (previously the wrapped 2**64-1 and 2**64-2).
   Non-negative results keep uint64
   (max(uint64[2**63+5], True) - uint64[3] -> 2**63+2). add/mul with
   negative literals keep the established modular semantics.
   Tests: test_literal_minus_uint64_exact,
   test_literal_minus_uint64_strict, test_uint64_minus_uint64_exact,
   test_uint64_minus_uint64_strict,
   test_sub_nonnegative_result_keeps_uint64,
   test_sub_mixed_magnitude_fallback_documented.

Round-24 regressions: tests/test_review_round24.py (6 tests). make
check green: 633 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged
(helper-internal logic only).
