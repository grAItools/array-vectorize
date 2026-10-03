# Open review findings — round 7 (reviewer agent 94e52ea7, reviewed commit 1bcb0b6)

STATUS: both round-7 findings fixed; round-8 request sent. Fix summary:

1. CRITICAL identical-dtype fast path compared promotion CLASSES, not
   actual dtypes — FIXED. The fast path now requires every ARRAY argument
   to share one ACTUAL dtype (dtype equality, not (class, width)) AND
   every raw literal to be EXACTLY representable in it (a new
   `_fits_dtype` check: IEEE round trip at the dtype's width for floats
   — float32 cannot hold 16777217 — and range checks for ints; bool
   literals only fit bool dtypes). uint8/int16 arrays now promote through
   the class-based slow path to int16 (max -> 300, min -> -1); a bool
   array with the literal 2 promotes to int8 (max -> 2); uint64 with a
   float64 array promotes to float64 (min -> 1.5).
   Tests: test_max_uint8_int16_no_narrowing, test_min_uint8_int16_negative,
   test_max_bool_array_vs_literal, test_min_uint64_vs_float64,
   test_min_float32_literal_needs_wider_dtype,
   test_min_float32_exact_literal_keeps_float32,
   test_min_int_literal_fit_boundaries.

2. CRITICAL uint64-as-float64 rounded selected integer results — FIXED by
   the same literal-fit fast path: when every array argument is uint64
   and the literal fits uint64 exactly (e.g. 1), the dtype stays uint64
   and the result is exact (max(9223372036854775809, 1) ->
   9223372036854775809). float64 is only used when promotion actually
   requires it (mixed-kind arguments or non-fitting literals).
   Tests: test_max_uint64_literal_exact.

Round-7 regressions: tests/test_review_round7.py (8 tests). make check
green: 527 tests, ruff + mypy strict clean, 95% branch coverage. Fuzzer
clean on seeds 42/7/123/999/2024. Goldens unchanged (helper-internal
logic only).
