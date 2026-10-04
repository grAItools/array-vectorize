# Open review findings — round 19 (reviewer agent 94e52ea7, reviewed commit 0d99838)

STATUS: all three round-19 findings fixed; round-20 request sent. Fix
summary:

1. CRITICAL incorrect remainder formula for negative divisors — FIXED.
   `a % -|d|` now computes `0` when `a %% |d| == 0`, else
   `(a %% |d|) - |d|` (in int64): 5 % -3 -> -1, 6 % -3 -> 0 (previously
   -(5 % 3) == -2). The -2 tests kept passing because |2|-1 == 1.
   Tests: test_uint64_mod_negative_divisor_formula.

2. MAJOR floor-division corrections added booleans to numeric arrays —
   FIXED. The ceil corrections cast the boolean comparison to the
   numeric operand's dtype before adding: max(x, True) // -3 -> -2 and
   -3 // max(x, True) -> -1 on array_api_strict (previously TypeError).
   Tests: test_uint64_floordiv_negative_divisor_strict,
   test_uint64_negative_literal_floordiv_strict.

3. CRITICAL literal-left remainder discarded representable uint64
   results — FIXED. The literal-left remainder result is always in
   [0, v) and exactly representable in uint64, so it is returned as the
   uint64 computation itself (no int64/float64 conversion):
   -1 % (2**63 + 2) -> 9223372036854775809 exactly.
   Tests: test_negative_literal_mod_uint64_exact,
   test_negative_literal_mod_small_array.

Round-19 regressions: tests/test_review_round19.py (5 tests). make
check green: 606 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged
(helper-internal formulas only).
