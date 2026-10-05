"""Statement lowering: assignments and if/elif/else branches.

Assignments allocate SSA names and record kind/literal/scalar facts;
``if`` lowers both branches from the pre-state, merges the resulting
environments with where-selects, and routes divergent early returns
through the deferred-return machinery.
"""

from __future__ import annotations

import ast

from ..ir import Binding, Literal, Logical, Node, UnaryOp, Where, is_bool
from .expressions import _ExpressionLowerer

__all__ = ["_StatementLowerer"]


class _StatementLowerer(_ExpressionLowerer):
    # --------------------------------------------------------- statements

    def lower_assign(self, stmt: ast.Assign | ast.AugAssign) -> None:
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            assert isinstance(target, ast.Name)  # validator guarantees
            value = self.lower_expr(stmt.value)
        else:
            target = stmt.target
            assert isinstance(target, ast.Name)
            value = self._lower_binop(stmt.op, self.load_name(target), self.lower_expr(stmt.value))
        var = target.id
        if var in self.loop_vars:
            raise self.error(stmt, f"cannot assign the loop variable {var!r} inside its loop")
        name: str | None = None
        if self.active_carried:
            carried, entry_depth = self.active_carried[-1]
            if var in carried and self.branch_depth == entry_depth:
                # loop-carried by the innermost active loop, assigned at the
                # loop body's own level: write the loop name directly.
                # (Inside nested branches, normal suffixing + merge + the
                # post-statement sync keep the loop name consistent.)
                name = carried[var]
        if name is None:
            name = self.ssa.bind(var, force_suffix=self.branch_depth > 0)
        self.bindings.append(Binding(name, value))
        self.definite[var] = name
        kind = self._numeric_kind(value)
        self.kinds.set_kind(name, kind)
        if isinstance(value, Literal):
            self.kinds.set_literal(name, value)
        else:
            self.kinds.drop_literal(name)
        if self._possibly_scalar(value):
            self.kinds.mark_scalar(name)
        else:
            self.kinds.unmark_scalar(name)
        if self._maybe_bool_result(value):
            self.kinds.mark_maybe_bool(name)
        else:
            self.kinds.unmark_maybe_bool(name)
        if self.active_carried:
            for carried, _depth in self.active_carried:
                if var in carried:
                    # a carried variable's runtime kind is the union of its
                    # phi kind and every body assignment; record this
                    # assignment's kind in every active loop frame
                    self.kinds.record_carried_kind(carried[var], kind)
        self.maybe.discard(var)

    def lower_if(self, stmt: ast.If) -> Node | None:
        cond = self.lower_expr(stmt.test)
        if not is_bool(cond):
            raise self.error(
                stmt,
                "bare truthiness in an if condition is ambiguous per-lane; "
                "write an explicit comparison, e.g. 'if x != 0:'",
            )
        pre_definite = dict(self.definite)
        pre_maybe = set(self.maybe)
        pre_deferred = list(self.deferred)

        # THEN branch
        self.branch_depth += 1
        then_result = self.lower_stmts(stmt.body)
        self.branch_depth -= 1
        then_definite, then_maybe = dict(self.definite), set(self.maybe)
        then_deferred = self.deferred

        # ELSE branch: reset to pre-state (bindings stay; both are emitted)
        self.definite, self.maybe = dict(pre_definite), set(pre_maybe)
        self.deferred = list(pre_deferred)
        self.branch_depth += 1
        else_result = self.lower_stmts(stmt.orelse)
        self.branch_depth -= 1
        else_definite, else_maybe = dict(self.definite), set(self.maybe)
        else_deferred = self.deferred
        pre_len = len(pre_deferred)

        if then_result is not None and else_result is not None:
            # Both branches return: this if terminates the block. Branch folds
            # applied their own entries; re-fold the merged value to apply
            # entries pending at this block's depth (they fire earlier in
            # program order, so they wrap the merged where from outside).
            self.deferred = list(pre_deferred)
            merged: Node = Where(cond, then_result, else_result)
            return self.fold_returns(merged)

        if then_result is not None:
            # then returns, else falls through: continue on the else path.
            # The if's own divergence encloses the else branch's pending
            # entries, so it is inserted before them (but after entries
            # pending from before this if).
            self.definite, self.maybe = else_definite, else_maybe
            self.deferred = [
                *else_deferred[:pre_len],
                (cond, then_result, self.branch_depth),
                *else_deferred[pre_len:],
            ]
            self._mark_then_only_unbound(pre_definite, then_definite, else_definite)
            return None
        if else_result is not None:
            # else returns, then falls through: continue on the then path.
            self.definite, self.maybe = then_definite, then_maybe
            not_cond = UnaryOp("not", cond)
            self.deferred = [
                *then_deferred[:pre_len],
                (not_cond, else_result, self.branch_depth),
                *then_deferred[pre_len:],
            ]
            self._mark_then_only_unbound(pre_definite, else_definite, then_definite)
            return None

        # both fall through: merge environments with where-selects
        self._merge_envs(cond, pre_definite, then_definite, else_definite, then_maybe, else_maybe)
        # qualify branch-pending returns with their branch conditions
        not_cond = UnaryOp("not", cond)
        qualified: list[tuple[Node, Node, int]] = []
        for c, v, p in then_deferred[pre_len:]:
            qualified.append((Logical("and", (cond, c)), v, p))
        for c, v, p in else_deferred[pre_len:]:
            qualified.append((Logical("and", (not_cond, c)), v, p))
        self.deferred = [*pre_deferred, *qualified]
        return None
