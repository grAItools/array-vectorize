# Open review findings — round 6 (reviewer agent 94e52ea7, reviewed commit 8e6987a)

STATUS: all round-6 findings fixed; round-7 request sent. Fix summary:

The promotion helpers were redesigned into two mode-specific runtime
helpers (both still injected through the generated module's namespace):

- `_vec_common_dtype(xp, *args)` — min/max mode: results are always one
  of the input values, so no arithmetic headroom is needed. Identical
  argument dtypes pass through unchanged; mixed int/float promotes to
  float64; ints widen to the largest width present; uint64 counts as
  float64 (no signed int holds it).
- `_vec_arith_dtype(xp, *args)` — bool-intify mode: Python ints are
  unbounded, so int-class operands get int64 headroom; float operands
  keep their dtype; mixed int/float promotes to float64 (Python
  semantics). The binop intify now casts BOTH operands to this dtype.

1. CRITICAL intify truncated float loop-mixed operands — FIXED. The
   arith helper receives the converted operand AND its sibling, so the
   cast target is a superset of the operand's actual runtime dtype (a
   float operand casts to float: no-op). Unary negation uses the operand
   alone (same rule), replacing the unconditional int64.
   Tests: test_mixed_loop_operand_not_truncated_by_literal,
   test_mixed_loop_operand_unary_negate.

2. CRITICAL int8 overflow for bool arithmetic — FIXED. The arith helper
   gives int-class operands int64 headroom (True + True chains and *100
   scaling cannot wrap).
   Tests: test_bool_arith_times_hundred, test_bool_loop_count_128
   (your 128-iteration counter).

3. CRITICAL uint64 promotion made positives negative — FIXED. uint64
   counts as float64 in the promotion lattice (no signed int holds it);
   min(x_uint64, 1) -> 1. Identical-dtype arguments (pure uint64 pairs)
   pass through with their own dtype, so no forced float64 rounding.
   Tests: test_min_uint64_vs_literal, test_minmax_identical_dtypes_passthrough.

4. CRITICAL float32 rank beat int64 precision — FIXED. Mixed int/float
   promotes to FLOAT64 (not the float's width), matching Python's
   int + float -> double: min(float32(20000000), int64(16777217)) ->
   16777217.
   Tests: test_min_float32_vs_int64_exact.

5. MAJOR helper name shadowed by user parameters — FIXED. Both helper
   names are allocated through the shared SSA env before parameters
   bind (and registered on the helpers list, which compile_vectorized
   injects), so a parameter named like a helper is renamed instead of
   shadowing the injected global.
   Tests: test_helper_name_shadowed_by_param,
   test_helper_names_shadowed_by_params.

6. MAJOR helper cache ignored protect_domains — FIXED. The cache key is
   (function, protect_domains): a protected compilation never reuses an
   unprotected helper.
   Tests: test_helper_cache_protect_flag (your exact repro, verified
   under errstate(all='raise')).

Round-6 regressions: tests/test_review_round6.py (11 tests). make check
green: 519 tests, ruff + mypy strict clean, 95% branch coverage. Fuzzer
clean on seeds 42/7/123/999/2024. Goldens regenerated.
