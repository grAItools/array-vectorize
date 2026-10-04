# Open review findings — round 10 (reviewer agent 94e52ea7, reviewed commit bc4ace3)

STATUS: the round-10 finding fixed; round-11 request sent. Fix summary:

1. CRITICAL integer-only min/max lost precision when uint64 promotion
   reached the slow path — FIXED by restructuring min/max into a single
   injected runtime helper `_vec_minmax(xp, is_min, *args)` that computes
   the RESULT with exact Python semantics instead of pre-casting to a
   lossy common dtype (replaces the astype + _vec_minmax_dtype +
   _vec_minmax_lit emission; the all-literal compile-time fold is kept).
   The selection strategy, in Python, from the runtime dtypes:
   - identical array dtypes with exactly-fitting (or never-winning,
     clamped) literal bounds keep that dtype;
   - mixed int/float promotes to float64; all-float uses the widest
     float dtype;
   - pure-integer mixes with uint64 use PROVEN result ranges: max
     results always fit uint64 (a signed value only wins a max when
     positive, so signed negatives clamp to 0 — and unsigned arrays
     widen to uint64); min results fit int64 whenever something signed
     can win (uint64 values above int64-max can never win a min against
     a signed value, so they clamp to int64-max);
   - other integer mixes widen to the containing signed dtype.
   minimum/maximum fold pairwise inside the helper (they are binary in
   the Array API).
   Repros: min(uint64[2**63+1], -9007199254740993) -> -9007199254740993
   exactly; min(x, -1) & 1 -> 1 (integer result dtype); max(uint64,
   int64[-1]) -> 2**63+1 and min -> -1 exactly.
   Tests: test_min_uint64_always_winning_literal_exact,
   test_min_uint64_result_supports_bitwise,
   test_max_uint64_int64_arrays_exact, test_min_uint64_int64_arrays_exact,
   test_min_uint64_signed_positive_literals,
   test_max_uint64_all_unsigned_widens, test_min_max_roundtrip_prior_findings,
   plus helper-branch coverage tests (fit-and-clamp literals, u64-max
   literals, float16-overflow promotion, _fits_dtype defensive branches).

Round-10 regressions: tests/test_review_round10.py (12 tests). make
check green: 551 tests, ruff + mypy strict clean, 95% branch coverage.
Fuzzer clean on seeds 42/7/123/999/2024. Goldens regenerated (min/max
now lowers to a single _vec_minmax helper call).
