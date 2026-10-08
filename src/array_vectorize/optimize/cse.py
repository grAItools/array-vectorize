# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Common-subexpression elimination over the IR."""

from __future__ import annotations

import collections

from array_vectorize import ir
from array_vectorize.ir import ssa
from array_vectorize.ir import walk

__all__ = ["cse"]

# ----------------------------------------------------------------------- CSE

_CSE_ELIGIBLE = (ir.Call, ir.Where)


def _count_eligible(node: ir.Node, counter: collections.Counter[ir.Node]) -> None:
    if isinstance(node, _CSE_ELIGIBLE):
        counter[node] += 1
    for child in walk.children(node):
        _count_eligible(child, counter)


def _has_loop(stmts: tuple[ir.Stmt, ...] | list[ir.Stmt]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, ir.Loop):
            if _has_loop(stmt.body):
                return True
            return True
    return False


def cse(program: ir.Program, ssa: ssa.SSAEnv) -> ir.Program:
    """Hoist duplicated Call/Where subtrees into temps ``t_1, t_2, ...``.

    Each pass counts eligible nodes in the *current* program and hoists those
    seen at least twice, inserting each temp just before the first binding
    that uses it (all bindings are at function scope, so hoisting is safe).
    Rebuilt parents (whose children became temps) only become countable in
    the next pass, hence the fixpoint loop.
    """
    current = program
    if _has_loop(list(program.bindings)):
        # CSE hoists temps before their first use, which is unsafe across
        # loop boundaries (zero-trip loops would skip the temp). Loops are
        # rare; skip CSE entirely for loop-containing programs (CSE is not
        # needed for correctness).
        return current
    for _ in range(8):
        counter: collections.Counter[ir.Node] = collections.Counter()
        for stmt in current.bindings:
            assert isinstance(stmt, ir.Binding)  # CSE skips loop programs
            _count_eligible(stmt.expr, counter)
        _count_eligible(current.result, counter)
        if not any(count >= 2 for count in counter.values()):
            return current

        memo: dict[ir.Node, str] = {}
        pending: list[ir.Binding] = []

        def rewrite(
            node: ir.Node,
            *,
            counter: collections.Counter[ir.Node] = counter,
            memo: dict[ir.Node, str] = memo,
            pending: list[ir.Binding] = pending,
        ) -> ir.Node:
            node = walk.rewrite(node, rewrite)
            if isinstance(node, _CSE_ELIGIBLE) and counter[node] >= 2:
                if node not in memo:
                    temp = ssa.fresh_temp("t")
                    pending.append(ir.Binding(temp, node))
                    memo[node] = temp
                return ir.Ref(memo[node])
            return node

        rewritten: list[ir.Stmt] = []
        for stmt in current.bindings:
            assert isinstance(stmt, ir.Binding)  # CSE skips loop programs
            start = len(pending)
            expr = rewrite(stmt.expr)
            rewritten.extend(pending[start:])
            rewritten.append(ir.Binding(stmt.name, expr))
        result_start = len(pending)
        result = rewrite(current.result)
        rewritten.extend(pending[result_start:])  # temps first used by the result go last
        current = ir.Program(current.params, tuple(rewritten), result)
    return current
