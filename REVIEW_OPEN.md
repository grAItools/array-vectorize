# Open review findings — round 4 (reviewer agent 94e52ea7, reviewed commit f3b6b12)

STATUS: all round-4 findings fixed; round-5 request sent. Fix summary:

1. CRITICAL float64 intify loses integer precision / breaks `&` — FIXED.
   intify now casts to int64 for pure bools (exact integer semantics,
   downstream `&` stays integer, large-int sums exact); float64 only for
   loop-carried names whose kind union includes float (`_float_mixed`),
   where int64 would truncate the float iterations. Bool literals intify
   to int literals.
   Tests: test_bool_plus_large_int_exact, test_bool_arith_then_bitwise.

2. CRITICAL two-pass cap misses chained kind propagation — FIXED. The
   loop body is re-lowered to a FIXED POINT: per-name labels only widen
   (int/float -> bool), so at most len(carried) widenings; the pass bound
   is len(carried) + 2 and the loop stops when computed labels equal the
   previous pass's. Also fixed the root cause that made mixes invisible:
   `_numeric_kind` now infers BinOp kinds (int/bool + float -> float,
   arithmetic over ints stays int, true division -> float) and UnaryOp
   neg/pos kinds.
   Tests: test_chained_loop_kind_mixing (your 4-iteration chain),
   test_float_bool_loop_mix_keeps_float_values.

3. CRITICAL min/max literal rounding with unknown sibling kind — FIXED.
   Literal args dtype-match their Ref sibling at RUNTIME via a new
   DTypeOf IR node: `minimum(x, astype(asarray(3), x.dtype))` (astype
   takes dtype positionally; DTypeOf codegens to `<value>.dtype`, with
   asarray-wrapping for possibly-scalar siblings). Exact for any integer
   width; no float64 rounding; strict promotion satisfied.
   Tests: test_min_literal_exact_large_int,
   test_min_literal_strict_int_input,
   test_min_literal_strict_float_default_param.

4. MAJOR preamble asarray(int default) broke scalar promotion — FIXED by
   reverting the param preamble. Operators keep raw scalars (scalar
   promotion works on every backend); scalars are handled only at xp.*
   call sites.
   Tests: test_int_default_scalar_promotion_strict.

5. MAJOR computed local scalars / loop-index expressions — FIXED.
   `_scalar_names` tracks names whose runtime value may be a raw Python
   scalar (parameters, loop variables, literal bindings, and locals
   computed from them); the sanitizer asarray-wraps possibly-scalar Refs
   and compound expressions containing them (plus astype float64 for
   math calls).
   Tests: test_computed_local_scalar_math_strict,
   test_loop_index_math_expression_strict.

6. MAJOR protect_domains forward refs to loop-body bindings — FIXED. The
   rewrite-time availability set (`ahead`) now includes every name bound
   anywhere inside a loop body (recursively), not just the loop index, so
   a clamp can never reference a binding created inside a later loop.
   Tests: test_protect_domains_loop_body_binding_guard.

7. MAJOR all-literal min/max kept Ref nodes — FIXED. The all-literal path
   substitutes the actual Literal nodes (walrus-filtered), so
   const-folding fires and nothing reaches codegen as raw scalars.
   Tests: test_all_literal_minmax_refs_fold.

8. MAJOR verify crashed on constant-return functions — FIXED. verify_match
   broadcasts scalar / 0-d results to the input shape before comparing.
   Tests: test_verify_constant_return.

Round-4 regressions: tests/test_review_round4.py (13 tests). make check
green: 497 tests, ruff + mypy strict clean, 95% branch coverage. Fuzzer
clean on seeds 42/7/123/999/2024. Goldens regenerated.
