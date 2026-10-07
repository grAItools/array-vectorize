# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Dead-code elimination over the IR."""

from __future__ import annotations

from array_vectorize import ir
from array_vectorize.ir import walk

__all__ = ["dce"]

# ----------------------------------------------------------------------- DCE


def _mark_live(node: ir.Node, live: set[str]) -> None:
    if isinstance(node, ir.Ref):
        live.add(node.name)
    for child in walk.children(node):
        _mark_live(child, live)


def _mark_live_stmt(stmt: ir.Stmt, live: set[str]) -> ir.Stmt | None:
    """Mark liveness from a statement; returns the kept statement or None."""
    if isinstance(stmt, ir.Binding):
        if stmt.name in live:
            _mark_live(stmt.expr, live)
            return stmt
        return None
    # Loop: recompute liveness from the FULL body in reverse order until a
    # fixed point: a statement can be live only through a use appearing
    # EARLIER in the body (loop-carried feedback), which a single reverse
    # pass misses. Liveness only grows, so this terminates.
    kept: list[ir.Stmt] = []
    while True:
        before_live = set(live)
        new_kept: list[ir.Stmt] = []
        for inner in reversed(stmt.body):
            kept_inner = _mark_live_stmt(inner, live)
            if kept_inner is not None:
                new_kept.append(kept_inner)
        new_kept.reverse()
        kept = new_kept
        if live == before_live:
            break
    kept_any = len(kept) > 0
    _mark_live(stmt.start, live)
    _mark_live(stmt.stop, live)
    _mark_live(stmt.step, live)
    if kept_any or stmt.var in live:
        return ir.Loop(stmt.var, stmt.start, stmt.stop, stmt.step, tuple(kept))
    return None


def dce(program: ir.Program) -> ir.Program:
    """Drop bindings whose results are never used."""
    live: set[str] = set()
    _mark_live(program.result, live)
    kept: list[ir.Stmt] = []
    # single reverse pass suffices: SSA uses only earlier bindings
    for stmt in reversed(program.bindings):
        kept_stmt = _mark_live_stmt(stmt, live)
        if kept_stmt is not None:
            kept.append(kept_stmt)
    kept.reverse()
    return ir.Program(program.params, tuple(kept), program.result)
