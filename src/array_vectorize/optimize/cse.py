"""Common-subexpression elimination over the IR."""

from __future__ import annotations

from collections import Counter

from ..ir import Binding, Call, Loop, Node, Program, Ref, Stmt, Where, walk
from ..ir.ssa import SSAEnv
from ..ir.walk import children

__all__ = ["cse"]

# ----------------------------------------------------------------------- CSE

_CSE_ELIGIBLE = (Call, Where)


def _count_eligible(node: Node, counter: Counter[Node]) -> None:
    if isinstance(node, _CSE_ELIGIBLE):
        counter[node] += 1
    for child in children(node):
        _count_eligible(child, counter)


def _has_loop(stmts: tuple[Stmt, ...] | list[Stmt]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, Loop):
            if _has_loop(stmt.body):
                return True
            return True
    return False


def cse(program: Program, ssa: SSAEnv) -> Program:
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
        counter: Counter[Node] = Counter()
        for stmt in current.bindings:
            assert isinstance(stmt, Binding)  # CSE skips loop programs
            _count_eligible(stmt.expr, counter)
        _count_eligible(current.result, counter)
        if not any(count >= 2 for count in counter.values()):
            return current

        memo: dict[Node, str] = {}
        pending: list[Binding] = []

        def rewrite(
            node: Node,
            *,
            counter: Counter[Node] = counter,
            memo: dict[Node, str] = memo,
            pending: list[Binding] = pending,
        ) -> Node:
            node = walk.rewrite(node, rewrite)
            if isinstance(node, _CSE_ELIGIBLE) and counter[node] >= 2:
                if node not in memo:
                    temp = ssa.fresh_temp("t")
                    pending.append(Binding(temp, node))
                    memo[node] = temp
                return Ref(memo[node])
            return node

        rewritten: list[Stmt] = []
        for stmt in current.bindings:
            assert isinstance(stmt, Binding)  # CSE skips loop programs
            start = len(pending)
            expr = rewrite(stmt.expr)
            rewritten.extend(pending[start:])
            rewritten.append(Binding(stmt.name, expr))
        result_start = len(pending)
        result = rewrite(current.result)
        rewritten.extend(pending[result_start:])  # temps first used by the result go last
        current = Program(current.params, tuple(rewritten), result)
    return current
