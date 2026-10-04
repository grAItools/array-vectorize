# Open review findings — round 14 (reviewer agent 94e52ea7, reviewed commit 3b428af)

STATUS: all three round-14 findings fixed; round-15 request sent. Fix
summary:

1. CRITICAL branch merges lost maybe-boolean tracking — FIXED.
   `_maybe_bool_result` handles Where nodes (a branch merge of
   maybe-bool values is maybe-bool), and `_merge_envs` records the
   merged binding name. `if x == True: a = min(x, True) else:
   a = max(x, False); a + a` -> [2, 0].
   Tests: test_branch_merged_maybe_bool_arithmetic.

2. CRITICAL maybe-boolean conversion rounded uint64 values — FIXED.
   `_vec_arith_dtype` preserves uint64 when no real float is involved
   and every other operand is exactly representable in uint64
   (unsigned/boolean arrays, non-negative literals): the u64-as-float64
   lattice classification no longer routes integer arithmetic through
   float64. `a = max(x, True); a + 0` at uint64[2**63+1] ->
   9223372036854775809 exactly (checked via int()).
   Tests: test_maybe_bool_uint64_arithmetic_exact,
   test_maybe_bool_uint64_unary_negate_no_rounding_cast.

3. MAJOR unary negation ignored maybe-boolean results — FIXED. The
   USub intify condition now also fires for maybe-bool operands:
   `a = min(x, True); -a` -> [-1, 0] on numpy and array_api_strict.
   Tests: test_unary_negate_maybe_bool,
   test_unary_negate_maybe_bool_strict.

Round-14 regressions: tests/test_review_round14.py (5 tests). make
check green: 576 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens unchanged.
