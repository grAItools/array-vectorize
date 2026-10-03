# Open review findings — round 8 (reviewer agent 94e52ea7, reviewed commit dcafa78)

STATUS: both round-8 findings fixed; round-9 request sent. Fix summary:

1. CRITICAL float16 literals checked at float32 precision — FIXED.
   `_fits_dtype` maps the dtype's width to the matching IEEE format
   ({16: 'e', 32: 'f', 64: 'd'}); exotic widths are conservatively
   rejected (promote instead of risking rounding). A float16-unfit
   literal (2049.0) now fails the exact-fit check and promotes to
   float64: max(float16[0], 2049.0) -> 2049.0; a fitting literal
   (2048.0) keeps float16.
   Tests: test_max_float16_literal_precision,
   test_max_float16_exact_literal_keeps_dtype.

2. CRITICAL uint64 precision loss with a negative max-bound — FIXED with
   clamp-aware literal bounds. Literal min/max bounds now go through an
   injected `_vec_minmax_lit(xp, is_min, lit, *arrays)` helper: a bound
   that FITS the arrays' shared dtype is cast exactly; a bound that can
   NEVER WIN is CLAMPED into an unsigned dtype instead of forcing
   promotion (max with a negative bound -> 0; min with a bound >= 2**bits
   -> uint-max) — preserving the exact array values; anything else
   promotes (mixed int/float -> float64). The array-side common dtype
   (`_vec_minmax_dtype`, replacing _vec_common_dtype, now taking the
   is_min flag) applies the same rule, so clamped bounds keep the
   array's dtype. max(uint64[2**63+1], -1) -> 9223372036854775809
   exactly; min(uint64[...], -1) -> -1 (the bound wins and survives).
   Narrow unsigned dtypes clamp too: min(uint8, 300) -> 255-clamp,
   max(uint8, -1) -> 0-clamp.
   Tests: test_max_uint64_negative_bound_exact,
   test_min_uint64_negative_bound_literal_wins,
   test_min_uint8_too_large_bound_clamps,
   test_max_uint8_negative_bound_clamps.

Round-8 regressions: tests/test_review_round8.py (6 tests). make check
green: 533 tests, ruff + mypy strict clean, 95% branch coverage. Fuzzer
clean on seeds 42/7/123/999/2024. Goldens regenerated (min/max literal
args now lower through _vec_minmax_lit).
