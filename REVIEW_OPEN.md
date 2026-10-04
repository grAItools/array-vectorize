# Open review findings — round 16 (reviewer agent 94e52ea7, reviewed commit c2b7ba0)

STATUS: the round-16 finding fixed; round-17 request sent. Fix summary:

1. CRITICAL modular uint64 casting was applied to remainder and
   division — FIXED. `_vec_arith_dtype` now receives the OPERATOR (as a
   stable int id indexing _ARITH_OPS; generated code passes an int
   literal) and applies the uint64-preservation only where modular
   arithmetic preserves semantics:
   - add/sub/mul: any int literal (modular wrap is exact while the true
     result is in [0, 2**64) — all any dtype can hold);
   - floordiv/mod: only non-negative literals (negative divisors change
     floor/mod semantics: -2 % 3 == 1 and 3 // -2 == -2, so those take
     the lattice path, which promotes u64 to float64 and computes
     exactly for these cases);
   - true division and negation: never the uint64 path (results can be
     negative or float) — the operands are cast to float64, which also
     fixes the strict-backend rejection of integer true division.
   Repros: -2 % a -> 1; a // -2 -> -2; a / -2 -> -1.5 on numpy AND
   array_api_strict; a // 2 and a % 2 keep the exact modular uint64
   path for non-negative divisors; a + -2 stays modular-exact.
   Tests: test_uint64_mod_negative_divisor,
   test_uint64_floordiv_negative_divisor,
   test_uint64_truediv_negative_divisor,
   test_uint64_truediv_negative_divisor_strict,
   test_uint64_floordiv_mod_positive_keep_uint64,
   test_uint64_add_negative_literal_still_modular.

Round-16 regressions: tests/test_review_round16.py (6 tests). make
check green: 587 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens regenerated (the
arith helper calls now carry the operator id).
