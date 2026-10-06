# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release: source-to-source array vectorization for the
[Python Array API](https://data-apis.org/array-api/latest/).

### Added

- `vectorize(f)` compiles an inspectable scalar Python function into a
  semantically equivalent function over arrays, runnable unchanged on
  NumPy, PyTorch, JAX, CuPy, `array-api-strict`, and any other
  standard-compliant backend. The supported scalar subset covers
  `if`/`elif`/`else` with early returns, constant-trip `for i in
  range(N)` loops, arithmetic and comparisons (including mixed-dtype
  and boolean operands), `and`/`or`/`not`, `min`/`max`, `abs`/`round`,
  numeric casts, `math` functions, calls to other vectorizable
  functions, and scalar/array closures.
- Exactness contract: anything outside the supported subset raises
  `VectorizationError` with all violations collected at once and
  precise line/column diagnostics — never a silent miscompile. Integer
  and boolean edge semantics (mixed kinds, uint64 values above 2^63)
  are preserved through runtime helpers; the few inherent divergences
  (numeric exceptions become IEEE values, eagerly evaluated branches)
  are documented in `docs/semantics.md`.
- Options: `strict=False`/`fallback=True` (element-loop fallback with
  a `UserWarning`), `protect_domains=True` (clamp partial-function
  arguments on provably dead lanes), `verify=example_args`
  (differential check against the scalar original at generation time),
  and `namespace=xp` / `vec.with_namespace(xp)` (pinned backends).
- Inspectability: `vec.source` and `inspect.getsource` show the
  generated function; tracebacks point at its real lines; the
  signature, name, and module are preserved; the docstring carries the
  original's documentation (summary prefixed `(array-vectorized)`)
  with the scalar source in a `Notes:` section; and the scalar
  original grows an `__array_vectorized__` back-reference.
- Grammar fuzzer (`python -m array_vectorize.fuzz`) and a differential
  Hypothesis test suite comparing generated functions against scalar
  Python on edge values.

[0.1.0]: https://github.com/grAItools/array-vectorize/releases/tag/v0.1.0
