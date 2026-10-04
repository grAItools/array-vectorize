# Open review findings — round 25 (reviewer agent 94e52ea7, reviewed commit 882d4b5)

STATUS: the round-25 finding fixed; round-26 request sent. Fix summary:

1. CRITICAL subtraction inferred result ranges after wrapping negative
   literals — FIXED. `_u64_sub_exact` now reads the true sign of
   Python int operands BEFORE any modular conversion (arrays on these
   paths are non-negative by construction: uint64/bool, or signed
   arrays gated non-negative at runtime):
   - right negative literal: left - (negative) = left + |right| is
     always non-negative -> modular uint64 (overflow past 2**64 is
     the documented backend behavior, same as add);
   - left negative literal: (negative) - right = -(|left| + right) is
     always negative -> exact int64 when the magnitude fits
     (runtime-checked, |left| + right <= 2**63, includes the
     int64-min edge); otherwise the documented float64;
   - both non-negative: the round-23/24 ge/pos/neg analysis unchanged.
   Repros: max(uint64[2**63+3], True) - -2 -> 2**63+5 (uint64, exact)
   and -2 - max(uint64[5], True) -> -7 (int64, exact), on numpy AND
   array_api_strict (previously -9223372036854775803 and the wrapped
   2**64-7). Also exact: closure constant -(2**63-1) minus the
   boolean max -> int64-min; one magnitude further falls back to the
   documented float64.
   Tests: test_uint64_minus_negative_literal_exact,
   test_uint64_minus_negative_literal_strict,
   test_negative_literal_minus_uint64_exact,
   test_negative_literal_minus_uint64_strict,
   test_negative_literal_sub_int64_min_exact,
   test_negative_literal_sub_past_int64_min_fallback.

Round-25 regressions: tests/test_review_round25.py (6 tests). make
check green: 639 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged
(helper-internal logic only).
