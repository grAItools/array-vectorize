---
name: add-construct
description: Add a scalar Python construct (operator, builtin, math function, statement form) to the supported subset of vectorize. Use when extending what vectorize accepts or turning a rejection into a lowering.
---

# Add a supported construct

Work test-first. Each step ends on its completion criterion; finish the
step before starting the next one.

1. **Pin the semantics.** Write a table of scalar Python results next to the
   Array API standard's specification for the candidate lowering, over the
   edge values `0`, `-0.0`, `±1`, subnormals, `±inf`, `NaN`, `True`/`False`,
   the int64 bounds, and uint64 values above `2**63` when integers apply.
   Pick one outcome: exact lowering, exact through a runtime helper,
   documented divergence (only if it is inherent to lane-parallel
   evaluation, like the cases in `docs/semantics.md`), or rejection.
   *Done when* every edge value has an expected vectorized result.

2. **Go red.** Add behavior tests asserting that the vectorized result
   equals the scalar loop: NumPy in `tests/behavior/test_<topic>.py` and
   `array_api_strict` in `tests/backends/test_array_api_strict.py`. Cover
   the edge table from step 1, including dtypes. *Done when* the tests fail
   because of the current rejection (`VectorizationError`), not because of
   a typo.

3. **Accept it.** Allow the node in `frontend/validate.py`. Name mappings
   for `math.*` and builtins live in `frontend/tables.py`. Every rejection
   that remains keeps a positioned diagnostic.

4. **Lower it.** Expressions go in `lower/expressions.py`, statements in
   `lower/statements.py`, and dtype/kind inference in `lower/kinds.py`.
   - Exactness that bare `xp.*` calls cannot give belongs in a runtime
     helper: add it under `runtime/` and register it in
     `runtime/registry.py`.
   - Foldable on literals: teach `optimize/constfold.py` only results that
     match the backend bit for bit.
   - Partial function (domain-restricted): add its domain to
     `_PARTIAL_DOMAINS` in `optimize/protect_domains.py`.

   *Done when* step 2's tests pass and `make lint type test` is green.

5. **Pin the output.** Add a representative function to `tests/corpus.py`
   and its name to `GOLDEN_NAMES`. Run `uv run pytest --update-golden` and
   read the new file in `tests/golden/cases/`: it should look like Array
   API code a person would write. If values are numeric, add a Hypothesis
   test in `tests/differential/test_differential.py`. Add unit tests for any
   new lowering or optimizer branch so coverage stays at or above 95%.

6. **Fuzz it.** If the construct is expression-level, add it to the grammar
   lists in `src/array_vectorize/fuzz.py` (`_UNARY_MATH`, `_BINOPS`, ...).
   *Done when* `make fuzz` is clean on all five seeds.

7. **Document it.** Add the row to `docs/supported-subset.md` and remove it
   from "Not supported". If step 1 chose a divergence, add it to the table
   in `docs/semantics.md`.

8. **Gate.** Run `make check`, `make docs`, and also `make backends` if the
   shape of the generated code changed. *Done when* all of them pass.
