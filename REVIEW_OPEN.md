# Open review findings — round 5 (reviewer agent 94e52ea7, reviewed commit 348606c)

STATUS: all round-5 findings fixed; round-6 request sent. Fix summary:

1. CRITICAL min/max literal truncation/overflow — FIXED. The DTypeOf
   sibling-matching is replaced by a runtime promotion helper
   `_vec_common_dtype(xp, *args)` injected into every generated module:
   it computes the common dtype with Python semantics (mixed int/float
   -> float; ints widen to fit: int8 -> int16 -> int32 -> int64), and
   every min/max argument is cast to it. A float bound widens the common
   dtype instead of truncating (min(x, 1.5) at int input -> 1.5); an int
   bound widens narrow sibling dtypes instead of overflowing
   (max(x, 300) at int8 -> 300).
   Tests: test_min_float_literal_int_input, test_max_literal_int8_no_overflow.

2. CRITICAL compound min/max siblings — FIXED. The helper receives ALL
   non-literal arguments (compound expressions included), so the common
   dtype is computed from the computed values themselves; big-int
   literals stay exact (CSE dedupes the repeated subexpressions).
   Tests: test_min_compound_sibling_exact_big_int.

3. CRITICAL float-mix lost through assignments — FIXED (and simplified
   away). `_float_mixed` static tracking is REMOVED entirely: the intify
   now casts the bool operand to the sibling's RUNTIME dtype via the
   same `_vec_common_dtype` helper, so loop-mixed operands (int in one
   iteration, float in another, bool in another) are cast to whatever
   dtype they actually have — never truncated.
   Tests: test_float_mixed_propagates_through_assignment.

4. CRITICAL true-division kind inference — FIXED. `_numeric_kind` now
   returns 'float' for ANY true division (Python int / int -> float),
   so `b = x / 2.0` records float in the loop-kind union.
   Tests: test_true_division_loop_mix_keeps_float.

5. CRITICAL bool bitwise classified int — FIXED. Bitwise and/or/xor of
   two bool-kind operands is 'bool' (Python bool & bool is bool); int
   involvement stays int.
   Tests: test_bool_bitwise_stays_bool, test_bool_bitwise_or_xor_stay_bool.

6. MAJOR bool + float-array on strict — FIXED by the same runtime-dtype
   intify: `(x > 0) + x` casts the bool to x's actual dtype (float64),
   so the operation stays single-dtype (strict rejects int64/float64
   array promotion). Works for float arrays, int arrays, literals (the
   helper widens to fit), and bool siblings (int8, exact for True+True).
   Tests: test_bool_plus_float_strict, test_bool_plus_float_literal_strict.

7. MAJOR scalar tracking through loop phis — FIXED. Loop phis inherit
   `_scalar_names` from their pre-loop binding (a raw scalar stays a raw
   scalar on zero-trip loops), so call-site sanitization wraps it.
   Tests: test_scalar_phi_strict.

Round-5 regressions: tests/test_review_round5.py (11 tests). make check
green: 508 tests, ruff + mypy strict clean, 95% branch coverage. Fuzzer
clean on seeds 42/7/123/999/2024. Goldens regenerated.
