# Open review findings — round 11 (reviewer agent 94e52ea7, reviewed commit c29b832)

STATUS: both round-11 findings fixed; round-12 request sent. Fix summary:

1. CRITICAL boolean min/max results bypassed boolean-arithmetic
   conversion — FIXED. `_numeric_kind` now infers `_vec_minmax` result
   kinds (bool when every argument is bool — seeing through asarray
   wrappers, which also preserve the wrapped kind; float when any
   argument is float; int for int-ish mixes), so downstream arithmetic
   intifies exactly: `a = min(x > 0, x > 1); a + a` -> 2, not True.
   Tests: test_bool_minmax_arithmetic_exact,
   test_bool_minmax_mixed_arithmetic, test_bool_minmax_bitwise_downstream.

2. MAJOR the helper used operations unsupported for boolean arrays on
   strict backends — FIXED. `_vec_minmax` normalizes boolean arrays to
   int8 at entry (0/1 exact for every boolean selection): the
   identical-dtype fast path no longer passes bools to xp.minimum /
   xp.maximum, and the uint64 signed-clamp comparisons never see bool
   operands. min(x > 0, x > 1) on array_api_strict -> [True, False]
   values; min/max(uint64, bool) mix works exactly.
   Tests: test_bool_minmax_strict, test_bool_minmax_strict_max,
   test_uint64_bool_mix_minmax, test_bool_minmax_with_literals.

Round-11 regressions: tests/test_review_round11.py (7 tests). make
check green: 558 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged (kind
inference and helper-internal normalization only).
