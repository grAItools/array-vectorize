# Supported subset

Anything outside this subset is **rejected with a precise diagnostic**
(line/column + caret) — never silently miscompiled.

| Scalar construct | Vectorized equivalent |
|---|---|
| `+ - * /` (and unary `- +`) | same operators |
| `x ** y` | `xp.pow(x, y)` |
| `x // y`, `x % y` | `xp.floor_divide`, `xp.remainder` (Python sign semantics) |
| `& \| ^ << >> ~` | same operators (integer inputs) |
| `abs`, `round`, `min`, `max` | `xp.abs`, `xp.round`, exact left-folded `xp.minimum`/`xp.maximum` |
| `int(x)`, `float(x)`, `bool(x)` | `xp.astype(x, xp.int64/float64/bool)` (truncation toward zero) |
| `math.sqrt exp expm1 log log1p log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh pow floor ceil hypot copysign isnan isinf isfinite` | corresponding `xp.*` |
| `math.trunc(x)` | `xp.astype(x, xp.int64)` |
| `math.pi/e/tau/inf/nan` | literals (`xp.inf` / `xp.nan` for the last two) |
| comparisons, chains (`a < b < c`) | elementwise; chains fold with `xp.logical_and` |
| `and` / `or` / `not` | exact value-select lowering (`xp.where(x != 0, y, x)`); `xp.logical_*` when both operands are boolean |
| ternary `a if c else b` | `xp.where(coerce(c), a, b)` |
| `if/elif/else`, early `return` | nested `xp.where` merges |
| assignment, `+=` etc. | SSA rebinding (`x`, `x_1`, `x_2`, ...) |
| `for i in range(N)` (N constant) | a real loop in the generated source; loop-carried variables become explicit phis |
| call to another pure scalar function | recursively vectorized, memoized by function object |
| closure/global scalars | frozen as constants at generation time |
| closure/global arrays | hidden keyword parameters with bound defaults |

## Not supported (rejected)

`while`, `break`/`continue`, `return` inside loop bodies, side effects,
comprehensions/generators, `*args`/`**kwargs`, `try`, `match`, walrus,
f-strings, subscripts, tuple/list/set/dict literals, tuple returns,
`global`/`nonlocal`, statements after `return`, annotated assignments,
non-literal defaults, `str`/`None`/`complex` literals, bare `if x:`
truthiness (write `if x != 0:`), recursive helper calls, and
`round(x, n)` (`decimals=` is not in the Array API standard).

## Exactness in the numeric edges

`min`/`max` over mixed operands and boolean arithmetic do not lower to
bare `xp.minimum`/operator calls: strict backends reject mixed-dtype
promotion and boolean arithmetic, and naive casts lose exactness.
Instead the lowering emits small runtime helpers
([architecture](architecture.md#runtime-helpers)) that compute exact
results whenever a single dtype can hold them, and produce a documented
[float64 fallback](semantics.md#documented-divergences) only for
genuinely unrepresentable per-lane results.
