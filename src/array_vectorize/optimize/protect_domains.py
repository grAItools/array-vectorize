# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""protect_domains: clamp partial-function arguments on dead lanes (design D1)."""

from __future__ import annotations

from array_vectorize import ir
from array_vectorize.ir import walk
from array_vectorize.ir import walk as walk_mod

__all__ = ["protect_domains"]

#: Partial functions: (lo, lo_open, hi, hi_open, safe_lo, safe_hi).
#: Protection clamps a partial call's argument ONLY on lanes that are dead
#: (discarded by the enclosing where-select) AND out of domain; live lanes
#: are never touched, so results never change (design D1). Open bounds
#: (poles such as log(0)) clamp dead lanes to an interior constant, so no
#: warnings are produced and float32 is safe.
_PARTIAL_DOMAINS: dict[str, tuple[float, bool, float | None, bool, float, float | None]] = {
    "sqrt": (0.0, False, None, False, 0.0, None),
    "log": (0.0, True, None, False, 1.0, None),
    "log2": (0.0, True, None, False, 1.0, None),
    "log10": (0.0, True, None, False, 1.0, None),
    "log1p": (-1.0, True, None, False, 0.0, None),
    "asin": (-1.0, False, 1.0, False, -1.0, 1.0),
    "acos": (-1.0, False, 1.0, False, -1.0, 1.0),
    "acosh": (1.0, False, None, False, 1.0, None),
    "atanh": (-1.0, True, 1.0, True, 0.0, 0.0),
}


def _and_ctx(ctx: ir.Node | None, cond: ir.Node) -> ir.Node:
    return cond if ctx is None else ir.Logical("and", (ctx, cond))


def _free_names(node: ir.Node) -> set[str]:
    """All Ref names appearing in a node tree."""
    names: set[str] = set()

    def walk(n: ir.Node) -> None:
        if isinstance(n, ir.Ref):
            names.add(n.name)
            return
        for child in walk_mod.children(n):
            walk(child)

    walk(node)
    return names


def _clamp_dead(
    arg: ir.Node,
    dead: ir.Node,
    spec: tuple[float, bool, float | None, bool, float, float | None],
) -> ir.Node:
    """Clamp ``arg`` to the domain, but only where ``dead`` holds."""
    lo, lo_open, hi, hi_open, safe_lo, safe_hi = spec
    clamped: ir.Node = arg
    if hi is not None:
        assert safe_hi is not None
        op = "ge" if hi_open else "gt"
        clamped = ir.Where(
            ir.Logical("and", (dead, ir.Compare(op, arg, ir.Literal(hi, "float")))),
            ir.Literal(safe_hi, "float"),
            clamped,
        )
    op = "le" if lo_open else "lt"
    clamped = ir.Where(
        ir.Logical("and", (dead, ir.Compare(op, arg, ir.Literal(lo, "float")))),
        ir.Literal(safe_lo, "float"),
        clamped,
    )
    return clamped


def protect_domains(program: ir.Program) -> ir.Program:
    """Clamp partial-function arguments on dead lanes only (design D1).

    A single reverse pass propagates liveness conditions: each binding's
    value is live under the disjunction of the conditions governing its uses
    (where-branch guards, conjunction-nested). A partial call's argument is
    clamped only where the value is dead AND out of domain, so live results
    never change and dead lanes never warn. Loop bodies are left unprotected
    (conservative) and their reads mark bindings fully live.
    """
    live: dict[str, ir.Node | None] = {}

    def or_live(name: str, ctx: ir.Node | None) -> None:
        if name not in live:
            live[name] = ctx
            return
        cur = live[name]
        if cur is None or ctx is None:
            live[name] = None
        else:
            live[name] = ir.Logical("or", (cur, ctx))

    def record_uses(node: ir.Node, ctx: ir.Node | None, skip: frozenset[str] = frozenset()) -> None:
        if isinstance(node, ir.Ref):
            if node.name not in skip:
                or_live(node.name, ctx)
        elif isinstance(node, ir.Where):
            record_uses(node.cond, ctx, skip)
            # the cond evaluates on every lane, so every name it reads is
            # already live under the full ctx; re-recording those names
            # under the narrowed branch ctx would only add redundant —
            # and possibly self-referential — disjuncts to their liveness
            cond_names = frozenset(n for n in _free_names(node.cond) if n not in skip)
            record_uses(node.then, _and_ctx(ctx, node.cond), skip | cond_names)
            record_uses(node.other, _and_ctx(ctx, ir.UnaryOp("not", node.cond)), skip | cond_names)
        else:
            for child in walk_mod.children(node):
                record_uses(child, ctx, skip)

    def rewrite(node: ir.Node, ctx: ir.Node | None) -> ir.Node:
        if isinstance(node, ir.Ref):
            return node
        if isinstance(node, ir.Where):
            cond = rewrite(node.cond, ctx)
            then = rewrite(node.then, _and_ctx(ctx, cond))
            other = rewrite(node.other, _and_ctx(ctx, ir.UnaryOp("not", cond)))
            return ir.Where(cond, then, other)
        if isinstance(node, ir.Call) and node.fn in _PARTIAL_DOMAINS:
            args = [rewrite(a, ctx) for a in node.args]
            if ctx is not None:
                dead = ir.UnaryOp("not", ctx)
                spec = _PARTIAL_DOMAINS[node.fn]
                args = [_clamp_dead(a, dead, spec) for a in args]
            return ir.Call(node.fn, tuple(args))
        return walk.rewrite(node, lambda n: rewrite(n, ctx))

    def mark_fully_live(node: ir.Node) -> None:
        if isinstance(node, ir.Ref):
            live[node.name] = None
        else:
            for child in walk_mod.children(node):
                mark_fully_live(child)

    def rewrite_stmt(stmt: ir.Stmt, ahead: set[str]) -> ir.Stmt:
        """Rewrite one binding.

        ``ahead`` holds the names bound AFTER this statement (they are
        not yet assigned where this statement runs).
        """
        if isinstance(stmt, ir.Binding):
            ctx = live.pop(stmt.name, None)
            record_ctx = ctx
            if ctx is not None and _free_names(ctx) & (ahead | {stmt.name}):
                # the liveness condition references names not bound at
                # this point (forward or self references), so it cannot be
                # evaluated here; skip the clamp (fully live) but keep the
                # liveness fact for upstream bindings
                ctx = None
            expr = rewrite(stmt.expr, ctx)
            record_uses(expr, record_ctx)
            ahead.add(stmt.name)
            return ir.Binding(stmt.name, expr)
        assert isinstance(stmt, ir.Loop)
        # conservative: loop bodies are not rewritten; every name they read
        # is marked fully live (later iterations may consume any binding)
        mark_fully_live(stmt.start)
        mark_fully_live(stmt.stop)
        mark_fully_live(stmt.step)
        for inner in stmt.body:
            if isinstance(inner, ir.Binding):
                mark_fully_live(inner.expr)
        # everything bound inside the loop (recursively) is not yet
        # assigned where statements BEFORE the loop run — loop-body
        # bindings count as forward references just like the index
        ahead.add(stmt.var)

        def collect_bound(st: ir.Stmt) -> None:
            if isinstance(st, ir.Binding):
                ahead.add(st.name)
            else:
                ahead.add(st.var)
                for inner in st.body:
                    collect_bound(inner)

        for inner in stmt.body:
            collect_bound(inner)
        return stmt

    new_result = rewrite(program.result, None)
    record_uses(new_result, None)
    ahead: set[str] = set()
    rewritten = [rewrite_stmt(stmt, ahead) for stmt in reversed(program.bindings)]
    rewritten.reverse()
    return ir.Program(program.params, tuple(rewritten), new_result)
