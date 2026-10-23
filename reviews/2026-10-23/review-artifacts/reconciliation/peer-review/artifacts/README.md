# Review reproduction artifacts

These scripts reproduce the verified findings in
[`../2026-10-23-architecture-and-exactness-review.md`](../2026-10-23-architecture-and-exactness-review.md).
They are evidence, not part of `make check`: each check states the scalar-Python
contract, prints the actual and expected result, and the script exits nonzero
while any defect remains. Once a finding is fixed, the script becomes a
regression harness.

Run from the repository root (the dev environment provides NumPy and
`array-api-strict`):

```bash
uv run python reviews/artifacts/repro_dtype_contract.py
uv run python reviews/artifacts/repro_backend_neutrality.py
uv run python reviews/artifacts/repro_emit_verify.py
```

| Script | Findings |
|---|---|
| `repro_dtype_contract.py` | F1 (bool/uint64/mixed arithmetic, merges, `_vec_arith` negative add/mul), F8, F9 |
| `repro_backend_neutrality.py` | F3, F7, F9 (zero divisors on integer arrays) |
| `repro_emit_verify.py` | F2, F5, F6, F8 (int64 minimum), F12 (`protect_domains` growth) |

Each script creates its subject modules in a temporary directory, because
`inspect.getsource` and the compiler's lambda identification require real
files.
