# Review evidence, 2026-10-23

These artifacts support [PROJECT_REVIEW.md](../../PROJECT_REVIEW.md), reviewing
commit `0e4bb6b64ae72ae369bbd61c921cf8673ec8d1a4`.

- `probes/` contains the exploratory scripts and generated probe subjects.
  Canonical repository license headers were prepended; their bodies are unchanged.
  These are archived review evidence, not maintained tests or proposed fixes.
- `results/` contains the numeric matrix results, validation logs, and benchmark
  JSON captured during the review. `make-check.log` records the gate rerun before
  committing this archive.
- `distributions/` contains the exact wheel and sdist built and smoke-tested during
  the review. They represent the reviewed commit, before this report/archive commit,
  and are not published releases.
- `MANIFEST.json` records original locations, original and archived SHA-256 hashes,
  and any transformation for the preserved artifacts. The gate rerun log is recorded
  separately as current validation rather than an original review artifact.

Run the consolidated compiler reproductions from the repository root:

```bash
uv run python review-artifacts/2026-10-23/probes/semantics_review_repros.py
uv run python review-artifacts/2026-10-23/probes/probe_helper_literal.py
uv run python review-artifacts/2026-10-23/probes/probe_project_api_review.py
uv run python review-artifacts/2026-10-23/probes/probe_project_quality_review.py
```

The scripts intentionally print incorrect outputs and caught exceptions as evidence;
exit status zero does not mean the compiler is correct. The additional Torch probe
needs the optional backend dependencies:

```bash
uv run --group backends python review-artifacts/2026-10-23/probes/probe_project_additional_review.py
```

`probe_matrix.py` and `probe_rejections.py` generate scratch subjects/results under
their original `/tmp` paths. Their archived generated files and original matrix
results are retained for comparison. Logs and benchmark JSON include the original
local paths and machine/environment metadata.
