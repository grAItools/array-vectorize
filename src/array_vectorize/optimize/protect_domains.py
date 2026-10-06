"""protect_domains: clamp partial-function arguments on dead lanes (design D1)."""

from __future__ import annotations

from array_vectorize.ir import (
    Binding,
    Call,
    Compare,
    Literal,
    Logical,
    Loop,
    Node,
    Program,
    Ref,
    Stmt,
    UnaryOp,
    Where,
    walk,
)
from array_vectorize.ir.walk import children

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


def _and_ctx(ctx: Node | None, cond: Node) -> Node:
    return cond if ctx is None else Logical("and", (ctx, cond))


def _free_names(node: Node) -> set[str]:
    """All Ref names appearing in a node tree."""
    names: set[str] = set()

    def walk(n: Node) -> None:
        if isinstance(n, Ref):
            names.add(n.name)
            return
        for child in children(n):
            walk(child)

    walk(node)
    return names


def _clamp_dead(
    arg: Node,
    dead: Node,
    spec: tuple[float, bool, float | None, bool, float, float | None],
) -> Node:
    """Clamp ``arg`` to the domain, but only where ``dead`` holds."""
    lo, lo_open, hi, hi_open, safe_lo, safe_hi = spec
    clamped: Node = arg
    if hi is not None:
        assert safe_hi is not None
        op = "ge" if hi_open else "gt"
        clamped = Where(
            Logical("and", (dead, Compare(op, arg, Literal(hi, "float")))),
            Literal(safe_hi, "float"),
            clamped,
        )
    op = "le" if lo_open else "lt"
    clamped = Where(
        Logical("and", (dead, Compare(op, arg, Literal(lo, "float")))),
        Literal(safe_lo, "float"),
        clamped,
    )
    return clamped


def protect_domains(program: Program) -> Program:
    """Clamp partial-function arguments on dead lanes only (design D1).

    A single reverse pass propagates liveness conditions: each binding's
    value is live under the disjunction of the conditions governing its uses
    (where-branch guards, conjunction-nested). A partial call's argument is
    clamped only where the value is dead AND out of domain, so live results
    never change and dead lanes never warn. Loop bodies are left unprotected
    (conservative) and their reads mark bindings fully live.
    """
    live: dict[str, Node | None] = {}

    def or_live(name: str, ctx: Node | None) -> None:
        if name not in live:
            live[name] = ctx
            return
        cur = live[name]
        if cur is None or ctx is None:
            live[name] = None
        else:
            live[name] = Logical("or", (cur, ctx))

    def record_uses(node: Node, ctx: Node | None, skip: frozenset[str] = frozenset()) -> None:
        if isinstance(node, Ref):
            if node.name not in skip:
                or_live(node.name, ctx)
        elif isinstance(node, Where):
            record_uses(node.cond, ctx, skip)
            # the cond evaluates on every lane, so every name it reads is
            # already live under the full ctx; re-recording those names
            # under the narrowed branch ctx would only add redundant —
            # and possibly self-referential — disjuncts to their liveness
            cond_names = frozenset(n for n in _free_names(node.cond) if n not in skip)
            record_uses(node.then, _and_ctx(ctx, node.cond), skip | cond_names)
            record_uses(node.other, _and_ctx(ctx, UnaryOp("not", node.cond)), skip | cond_names)
        else:
            for child in children(node):
                record_uses(child, ctx, skip)

    def rewrite(node: Node, ctx: Node | None) -> Node:
        if isinstance(node, Ref):
            return node
        if isinstance(node, Where):
            cond = rewrite(node.cond, ctx)
            then = rewrite(node.then, _and_ctx(ctx, cond))
            other = rewrite(node.other, _and_ctx(ctx, UnaryOp("not", cond)))
            return Where(cond, then, other)
        if isinstance(node, Call) and node.fn in _PARTIAL_DOMAINS:
            args = [rewrite(a, ctx) for a in node.args]
            if ctx is not None:
                dead = UnaryOp("not", ctx)
                spec = _PARTIAL_DOMAINS[node.fn]
                args = [_clamp_dead(a, dead, spec) for a in args]
            return Call(node.fn, tuple(args))
        return walk.rewrite(node, lambda n: rewrite(n, ctx))

    def mark_fully_live(node: Node) -> None:
        if isinstance(node, Ref):
            live[node.name] = None
        else:
            for child in children(node):
                mark_fully_live(child)

    def rewrite_stmt(stmt: Stmt, ahead: set[str]) -> Stmt:
        """Rewrite one binding; ``ahead`` holds the names bound AFTER this
        statement (they are not yet assigned where this statement runs)."""
        if isinstance(stmt, Binding):
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
            return Binding(stmt.name, expr)
        assert isinstance(stmt, Loop)
        # conservative: loop bodies are not rewritten; every name they read
        # is marked fully live (later iterations may consume any binding)
        mark_fully_live(stmt.start)
        mark_fully_live(stmt.stop)
        mark_fully_live(stmt.step)
        for inner in stmt.body:
            if isinstance(inner, Binding):
                mark_fully_live(inner.expr)
        # everything bound inside the loop (recursively) is not yet
        # assigned where statements BEFORE the loop run — loop-body
        # bindings count as forward references just like the index
        ahead.add(stmt.var)

        def collect_bound(st: Stmt) -> None:
            if isinstance(st, Binding):
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
    return Program(program.params, tuple(rewritten), new_result)
