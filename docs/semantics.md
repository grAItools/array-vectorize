# Semantics & divergences

## Semantics decisions

### Eager branches (dead lanes)

`if c: y = f(x) else: y = g(x)` lowers to `xp.where(c, f(x), g(x))`:
both branches evaluate on all lanes. Values are correct; NaN/warnings
from dead lanes are discarded by `where`.

`protect_domains=True` clamps partial-function arguments (`sqrt`, `log`,
`asin`, ...) on lanes that are provably dead (discarded by their
enclosing where-select) and out of domain — live lanes are never touched
(subnormals and signed zeros stay exact), warnings on dead lanes
disappear. Loop bodies and cross-function (helper) dead lanes are not
tracked (conservative).

### `and`/`or`/`not` are exact

Numeric operands lower to value-selects
(`x and y` → `xp.where(x != 0, y, x)`), including NaN truthiness
(`not nan` is `False` in both Python and `xp.logical_not`). When both
operands are boolean, `xp.logical_and/or` is emitted instead.

### Integer exactness lattice

Mixed-width integer arithmetic (uint64 arrays with int64 arrays, boolean
operands, negative literals) goes through runtime helpers that pick, per
call and checked at runtime on the actual values:

1. **int64** when every operand value fits int64 (handles negative
   results exactly; overflow beyond int64 is the documented backend
   behavior);
2. **modular uint64** when the result is provably non-negative
   (`add`/`sub`/`mul`), or `%`/`//` with non-negative dividends and
   divisors;
3. **result-aware subtraction** — uint64 when every per-lane difference
   is non-negative, else the exact mod-2^64 int64 difference when every
   difference fits int64 (including the int64-min edge);
4. **float64** only when per-lane results fit no single dtype (a huge
   uint64 plus a negative signed value, or mixed-magnitude differences).

### Reject over miscompile

Anything not provably translatable raises `VectorizationError` with all
violations collected at once.

## Documented divergences

Vectorization is lane-parallel, so a few scalar behaviors become IEEE
values. These are deliberate, tested, and inherent:

| Scalar behavior | Vectorized behavior |
|---|---|
| `ZeroDivisionError` from `1/x` | `inf`/`-inf` (IEEE) |
| `ZeroDivisionError` from `x % 0` | `NaN` |
| `(-8) ** 0.5` returns `complex` | `NaN` |
| `math` domain errors (`sqrt(-1)`) | `NaN` (plus a backend warning) |
| integer overflow in `+`/`*` folds | folding skipped; backend behavior |
| `min(a, nan)` returns `a` | `xp.minimum` propagates `NaN` |
| `round(-0.0)` returns int `0` (sign dropped) | `xp.round` keeps `-0.0` |
| branch-taken type kept (`0 if c else False`) | `xp.where` promotes branch dtypes |
| bool arithmetic is integer (`True + True == 2`) | exact: bool operands are cast to float64 (strict backends reject bool arithmetic) |
| scalar exceptions inside `verify=` | expected `NaN` lanes |
| mixed int/float `min`/`max` siblings | strict backends require matching array dtypes; literals adopt a provably-int sibling's dtype, else float64 |
| uint64 + signed arrays with mixed-sign per-lane results | float64 approximation (no single dtype holds both operand ranges) |
