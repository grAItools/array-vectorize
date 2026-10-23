# Reconciliation evidence

The canonical combined assessment is [RECONCILED_REVIEW.md](../../RECONCILED_REVIEW.md).
The compiler base is `0e4bb6b64ae72ae369bbd61c921cf8673ec8d1a4`.

- Reviewer A: Paseo `3ae67309-cf05-49d7-961e-c13ec5ea721e`; report and artifacts
  committed at `33efb77` under `PROJECT_REVIEW.md` and `review-artifacts/2026-10-23/`.
- Reviewer B: Paseo `3dd792b2-ca06-4084-911e-c74113895e1e`; independent review at
  `a531fe0`, copied unchanged under `peer-review/` for durable provenance.
- `probes/reconcile_additional_probes.py` contains Reviewer A's extra reproduction
  pass, including runtime-helper negatives, strict scalar boundaries, decorators,
  numeric limits, verifier behavior, callable fallback, and controlled concurrency.
  Its body is preserved with the canonical license header prepended.
- `results/` records reruns of the original peer scripts and the extra pass.
- `MANIFEST.json` records provenance and SHA-256 hashes for those preserved artifacts.
- `peer-reconciliation-response.md` preserves Reviewer B's agreement/correction memo.
- `results/make-check.log` records the successful reconciliation gate rerun:
  1,160 passed, 1 skipped, combined statement/branch coverage 96%.
- `CONSENSUS.md` records approval of the final report by exact SHA-256 digest.

Run the additional probe from the repository root:

```bash
uv run python review-artifacts/reconciliation/probes/reconcile_additional_probes.py
```

These exploratory probes print observations; exit zero is not a correctness verdict.
Original peer probes report `DEFECT` and exit nonzero for some expectations withdrawn
or reclassified during reconciliation. Their totals are historical output, not the
accepted finding count. In particular:

1. Mixed-type branch promotion is documented; the original negative-int64 merge
   has no common exact integer dtype.
2. Float64 `x + int(1e30)` matches scalar floating arithmetic. Its original
   integer-valued expected oracle was incorrect.
3. `x - 9223372036854775808` has a positive out-of-range AST literal. The correct
   signed-minimum probe uses `return -9223372036854775808`.
4. The verifier's NaN-only exception policy and current-scalar cache-hit comparison
   are documented limitations; integer/sign blind spots and complex-pow crashes
   are distinct tool defects.
5. Captured-array conversion and a 2x protection code-size threshold are not
   unconditional correctness requirements.

See the reconciled report's decision log and crosswalk before interpreting any raw
historical check. No implementation changes are included in these review artifacts.
