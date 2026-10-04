# Open review findings — round 18 (reviewer agent 94e52ea7, reviewed commit 5ec7b00)

STATUS: the round-18 finding fixed; round-19 request sent. Fix
summary:

1. CRITICAL one large uint64 lane forced lossy arithmetic for the whole
   batch — FIXED by making the arith helper compute the WHOLE operation
   (`_vec_arith` replaces `_vec_arith_dtype`; the lowering emits a
   single helper call for bool/maybe-bool arithmetic and negation).
   uint64 integer paths are now exact PER LANE:
   - array % negative-literal and array // negative-literal: the results
     ALWAYS fit int64 (remainder magnitude < |divisor| <= 2**63; floor
     magnitude <= 2**63), so the magnitudes are computed exactly in
     uint64 and negated into int64 — [2**53+1, 2**63+1] % -2 ->
     [-1, -1] and // -2 -> [-4503599627370497, -4611686018427387905]
     (your repros);
   - negative-literal % array: (v - |d| mod v) mod v computed exactly in
     uint64, int64 when every lane's result fits;
   - negative-literal // array: -ceil(|d|/v), exact in int64;
   - add/sub/mul (and non-negative floordiv/mod): modular uint64, with
     negative literals wrapped modularly (NumPy rejects out-of-bounds
     ints in asarray);
   - negation: int64 when values fit, else float64 (those results are
     genuinely unrepresentable per lane);
   - true division: always float operands.
   Tests: test_uint64_mod_negative_mixed_lanes_exact,
   test_uint64_floordiv_negative_mixed_lanes_exact,
   test_uint64_negate_mixed_lanes_best_effort,
   test_uint64_negative_literal_mod_array_mixed,
   test_uint64_negative_literal_floordiv_array,
   test_uint64_add_negative_literal_modular_still_exact,
   test_unary_negate_maybe_bool_after_restructure,
   test_truediv_bool_strict_after_restructure.

Round-18 regressions: tests/test_review_round18.py (8 tests). make
check green: 601 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens regenerated (bool /
maybe-bool arithmetic now lowers to single _vec_arith helper calls).
