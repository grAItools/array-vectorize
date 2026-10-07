"""Fuzzer smoke test: small, seeded, fast."""

from __future__ import annotations

from array_vectorize import fuzz


def test_fuzzer_smoke() -> None:
    # deterministic seed; any internal error or value mismatch fails nonzero
    assert fuzz.main(["--seed", "2024", "--cases", "40", "--seconds", "60"]) == 0


def test_fuzzer_detects_mismatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # sabotage the oracle to prove the fuzzer fails loudly on mismatch
    import array_vectorize.fuzz as fz

    # The imported binding is a monkeypatch seam, not an explicit module export.
    original = getattr(fz, "vectorize")  # noqa: B009

    def lying_vectorize(fn, **kwargs):  # type: ignore[no-untyped-def]
        vec = original(fn, **kwargs)

        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            import numpy as np

            return np.asarray(original_fn_result(vec, args))  # type: ignore[name-defined]

        return wrapper

    def original_fn_result(vec, args):  # type: ignore[no-untyped-def]
        return [v + 1000.0 for v in vec(*args)]

    monkeypatch.setattr(fz, "vectorize", lying_vectorize)
    assert fz.main(["--seed", "2024", "--cases", "5", "--seconds", "60"]) == 1
