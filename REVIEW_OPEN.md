# Open review findings — round 13 (reviewer agent 94e52ea7, reviewed commit aec0dd4)

STATUS: the round-13 finding fixed; round-14 request sent. Fix summary:

1. CRITICAL runtime-boolean min/max results bypassed arithmetic
   conversion when an operand's static kind was unknown — FIXED with
   maybe-bool tracking. A new `_maybe_bool_names` set records bindings
   whose value is a `_vec_minmax` result with unknown static kind
   (parameter operands: the runtime dtype may be boolean). The
   arithmetic-intify condition in `_lower_binop` now also fires for
   maybe-bool operands (direct FuncCall calls, Refs to maybe-bool
   bindings, and reference chains through assignments — propagated in
   `_lower_assign` and loop phis). The intify itself is
   runtime-polymorphic (_vec_arith_dtype): int64 for booleans, a no-op
   for numeric dtypes, so nothing changes for float/int inputs.
   Repro: `a = min(x, True); a + a` with a boolean array -> [2, 0] on
   numpy AND array_api_strict (previously [True, False] / TypeError).
   Tests: test_minmax_runtime_bool_arithmetic_numpy,
   test_minmax_runtime_bool_arithmetic_strict,
   test_minmax_runtime_bool_ref_chain, test_minmax_runtime_numeric_noop,
   test_minmax_runtime_bool_loop_carried,
   test_minmax_runtime_bool_direct_call.

Round-13 regressions: tests/test_review_round13.py (6 tests). make
check green: 571 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens regenerated (maybe-bool
arithmetic emits the runtime intify casts).
