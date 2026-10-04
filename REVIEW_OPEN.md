# Open review findings — round 12 (reviewer agent 94e52ea7, reviewed commit cd62990)

STATUS: both round-12 findings fixed; round-13 request sent. Fix summary:

1. CRITICAL boolean/integer min/max introduced int8 overflow downstream —
   FIXED. When every array operand was boolean (normalized to int8) and
   non-boolean operands are present, `_vec_minmax` selects in INT64:
   the compiler introduced the bool->int conversion, and Python's
   integer semantics are unbounded, so downstream arithmetic must not
   overflow the normalized int8 (float mixes still take the float
   paths). min(x > 0, 1) * 100 * 2 -> 200, not -56.
   Tests: test_bool_int_minmax_no_int8_overflow,
   test_bool_int_minmax_large_literal.

2. MAJOR boolean-only min/max returned int8, breaking boolean bitwise
   operations — FIXED. `_vec_minmax` detects boolean-only selections
   (every array boolean AND every literal boolean), computes in int8
   (strict backends reject bool operands in minimum/maximum), and
   RESTORES the boolean dtype on the result: min(x > 0, x > 1) &
   (x > 0) works on strict backends (True), and the static 'bool' kind
   inference matches the runtime result again.
   Tests: test_bool_only_minmax_bitwise_strict,
   test_bool_only_minmax_bitwise_numpy, test_bool_only_minmax_or_bitwise,
   test_bool_only_minmax_still_arithmetics_exact, test_bool_float_minmax.

Round-12 regressions: tests/test_review_round12.py (7 tests). make
check green: 565 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged
(helper-internal logic only).
