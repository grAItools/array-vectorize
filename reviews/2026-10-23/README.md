# Architecture and exactness review, 2026-10-23

Start with the [reconciled report](RECONCILED_REVIEW.md). Two independent reviewers
approved the same report digest; their [consensus receipt](review-artifacts/reconciliation/CONSENSUS.md)
records that approval. The report concerns compiler revision
`0e4bb6b64ae72ae369bbd61c921cf8673ec8d1a4`; it does not implement the proposed fixes.

The approved report is byte-for-byte unchanged from its original location.
The source reports, reproduction scripts, original observations, benchmark data,
validation logs and manifests remain alongside it as evidence. Source-archive
provenance retains original paths and commit IDs; the commands below use this
directory's current location.

## Reproduce the observations

Run from the repository root with the default development dependencies installed:

```bash
uv run python reviews/2026-10-23/review-artifacts/2026-10-23/probes/semantics_review_repros.py
uv run python reviews/2026-10-23/review-artifacts/2026-10-23/probes/probe_helper_literal.py
uv run python reviews/2026-10-23/review-artifacts/2026-10-23/probes/probe_project_api_review.py
uv run python reviews/2026-10-23/review-artifacts/2026-10-23/probes/probe_project_quality_review.py
uv run python reviews/2026-10-23/review-artifacts/reconciliation/probes/reconcile_additional_probes.py
```

The additional Torch precision probe needs the optional backend group:

```bash
uv run --group backends python reviews/2026-10-23/review-artifacts/2026-10-23/probes/probe_project_additional_review.py
```

These exploratory scripts print observed failures; zero exit status is not a
correctness verdict. The original peer scripts deliberately exit nonzero and
include some expectations subsequently withdrawn or qualified. Read the
[interpretation notes](review-artifacts/reconciliation/README.md) and the report's
reconciliation decisions before using their raw failure counts.

`probe_matrix.py` and `probe_rejections.py` write scratch subjects/results under
`/tmp`. Preserved logs include original local paths and environment metadata.

## Evidence map

- [Reconciled report](RECONCILED_REVIEW.md): canonical findings, proposals,
  severity, limitations, withdrawn claims and original-report crosswalk.
- [Reviewer A's original report](PROJECT_REVIEW.md) and
  [evidence index](review-artifacts/2026-10-23/README.md).
- [Reviewer B's original report](review-artifacts/reconciliation/peer-review/2026-10-23-architecture-and-exactness-review.md)
  and [reconciliation memo](review-artifacts/reconciliation/peer-reconciliation-response.md).
- [Reconciliation evidence index](review-artifacts/reconciliation/README.md),
  [checksum manifest](review-artifacts/reconciliation/MANIFEST.json),
  [final approval](review-artifacts/reconciliation/peer-approval.md), and
  [gate output](review-artifacts/reconciliation/results/make-check.log).

The archived wheel and sdist under `review-artifacts/2026-10-23/distributions/`
are the exact packages smoke-tested for the original reviewed revision. They are
historical build evidence, not published releases or packages containing fixes.
