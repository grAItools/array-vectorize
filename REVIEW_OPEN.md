# Open review findings — round 9 (reviewer agent 94e52ea7, reviewed commit b57f8bd)

STATUS: both round-9 findings fixed; round-10 request sent. Fix summary:

1. CRITICAL negative literals bypassed the literal helpers — FIXED.
   `_literal_value` now recognizes negated literals (`-1` lowers to
   UnaryOp(neg, Literal(1))), so negative bounds flow through
   _vec_minmax_lit and the never-winning-bound clamp fires:
   max(uint64[2**63+1], -1) -> 9223372036854775809 exactly. The
   round-8 regression assertions were also fixed to compare exact
   Python ints (numpy scalar == rounds floats, which made the old test
   pass falsely).
   Tests: test_max_uint64_negative_literal_exact,
   test_min_uint64_negative_literal_wins_exactly,
   test_negative_literal_other_contexts; round-8 uint64 tests now use
   int() comparisons.

2. MAJOR variadic min/max literals disagreed on the final dtype — FIXED.
   Each literal's _vec_minmax_lit call now receives ALL arguments (the
   other literals included), and the helper computes the final common
   dtype from everything before casting its own bound: a fitting bound
   still casts to float64 when another bound forces promotion.
   max(uint8[2], 1, 1.5) -> 2 on strict (all three operands float64).
   Tests: test_max_variadic_literals_strict,
   test_min_variadic_negative_and_float_literals,
   test_max_variadic_fitting_literals_keep_dtype.

Round-9 regressions: tests/test_review_round9.py (6 tests). make check
green: 539 tests, ruff + mypy strict clean, 95% branch coverage. Fuzzer
clean on seeds 42/7/123/999/2024. Goldens regenerated.
