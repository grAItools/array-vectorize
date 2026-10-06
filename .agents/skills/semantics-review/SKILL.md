---
name: semantics-review
description: Adversarial review that hunts miscompiles in array-vectorize, meaning accepted scalar functions whose vectorized results differ from scalar Python. Use after changes to frontend/, lower/, optimize/, codegen/ or runtime/, or when asked for a semantics review.
---

# Semantics review

Your job is to break `array-vectorize`. The contract under attack: for
every function `vectorize(f)` accepts, `vectorize(f)(*arrays)` equals
`[f(*lane) for lane in zip(*arrays)]` on every lane, in value, dtype kind,
NaN-ness, and sign of zero. The only exceptions are the divergences listed
in `docs/semantics.md`.

Review from a fresh context, such as a subagent when your harness has them,
so the reasoning behind the implementation does not anchor the review. The
review is read-only: write probes and repros to `/tmp`, and leave the
repository unchanged.

Scope: the change you were pointed at (`git diff <base>`), or the whole
compiler if none was given. Read `docs/architecture.md` and
`docs/semantics.md` first so you know which behavior is documented.

## Method

1. Read the code under review and list its **suspects**: dtype promotion,
   bool/int/uint64 mixing, literal binding, branch merges, loop phis,
   constant folding, CSE across branches, `protect_domains` clamps,
   namespace detection, helper calls, and closures.
2. For each suspect, write small scalar functions that the validator
   accepts, and probe them. Use the edge values `0`, `-0.0`, `±1`,
   subnormals, `±inf`, `NaN`, `True`/`False`, the int64 bounds, and uint64
   values above `2**63`, mixed across argument dtypes (bool, int64,
   uint64, float64, plus Python scalars). Run on NumPy and on
   `array_api_strict`.
3. Probe the rejection path too: unsupported input must raise
   `VectorizationError` with a position, never another exception or a
   silent acceptance.

Run probes with `uv run python /tmp/probe_<n>.py`. Functions must live in
real files because `vectorize` reads source with `inspect.getsource`.

## Bar

A **finding** is an accepted function, plus inputs, whose vectorized
result differs from the scalar loop outside the documented divergences, or
any crash other than `VectorizationError` at decoration or call time. Each
finding needs a minimal standalone repro script that you ran yourself, with
its output.

Keep probing until every suspect has been exercised with the full edge set
on both backends. Report how many boundary checks you ran.

## Report

- Findings, most severe first: what breaks, the repro (path + content),
  observed vs expected, and the suspect code location (`file:line`).
- Suspects you exercised that held.
- Verdict: `APPROVED` when there are no findings, otherwise `CHANGES
  REQUESTED`.
