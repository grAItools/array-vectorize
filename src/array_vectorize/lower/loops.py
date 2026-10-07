"""Loop lowering: constant-trip ``for`` loops with loop-carried phis.

A loop's carried variables (bound before the loop, reassigned in the
body) get fresh loop names with explicit phi bindings just before the
loop; body statements lower through the dispatcher, which synchronizes
rebound values back to the loop names. Body kinds reach a fixed point
over repeated lowering passes, widening mixed-kind carried variables so
the exactness mitigations fire inside the body.
"""

from __future__ import annotations

import ast

from array_vectorize import ir
from array_vectorize.lower import statements
from array_vectorize.optimize import constfold

__all__ = ["_LoopLowerer"]


class _LoopLowerer(statements._StatementLowerer):
    # ------------------------------------------------------------- loops

    def _const_int(self, node: ast.expr, what: str) -> ir.Node:
        folded = constfold._const_fold_expr(self.lower_expr(node))
        if (
            isinstance(folded, ir.Literal)
            and folded.kind == "int"
            and -(2**63) <= folded.value < 2**63
        ):
            return folded
        raise self.error(node, f"{what} must be constant ints known at generation time")

    def _loop_bounds(self, call: ast.Call) -> tuple[ir.Node, ir.Node, ir.Node]:
        args = list(call.args)
        bounds: list[ir.Node] = [self._const_int(a, "loop bound") for a in args]
        if len(bounds) == 1:
            start_n: ir.Node = ir.Literal(0, "int")
            stop_n: ir.Node = bounds[0]
            step_n: ir.Node = ir.Literal(1, "int")
        elif len(bounds) == 2:
            start_n, stop_n = bounds
            step_n = ir.Literal(1, "int")
        else:
            start_n, stop_n, step_n = bounds
        assert isinstance(step_n, ir.Literal)
        if step_n.value == 0:
            raise self.error(call, "range step cannot be zero")
        return start_n, stop_n, step_n

    def _body_bound_names(self, stmts: list[ast.stmt], *, include_for_targets: bool) -> set[str]:
        """Names bound in a statement list.

        With ``include_for_targets``: also the targets of NESTED ``for``
        statements - Python loop targets bind in the enclosing scope, so a
        nested loop rebinding an outer variable is a loop-carried dependency
        of the outer loop. Without: only ``=``/augmented assignments (used
        for the cannot-assign-the-loop-variable check).
        """
        bound: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    bound.add(stmt.target.id)
            elif isinstance(stmt, ast.If):
                bound |= self._body_bound_names(stmt.body, include_for_targets=include_for_targets)
                bound |= self._body_bound_names(
                    stmt.orelse, include_for_targets=include_for_targets
                )
            elif isinstance(stmt, ast.For):
                if include_for_targets and isinstance(stmt.target, ast.Name):
                    bound.add(stmt.target.id)
                bound |= self._body_bound_names(stmt.body, include_for_targets=include_for_targets)
        return bound

    def lower_for(self, stmt: ast.For) -> None:
        """Lower a constant-trip ``for i in range(...)`` (design D6).

        Loop-carried variables (bound before the loop and reassigned in the
        body) get a fresh loop name with an explicit phi binding
        ``s_loop = s_pre`` just before the loop; body assignments write the
        loop name directly, so the real generated loop feeds values across
        iterations. Constructs that rebind a carried variable (branch merges,
        inner loops) are synchronized back after each body statement.
        """
        iter_call = stmt.iter
        assert isinstance(iter_call, ast.Call)  # validator guarantees range(...)
        start, stop, step = self._loop_bounds(iter_call)

        assert isinstance(stmt.target, ast.Name)  # validator guarantees
        loop_var = stmt.target.id
        loop_name = self.ssa.bind(loop_var)

        pre_definite = dict(self.definite)
        body_assigned = self._body_bound_names(stmt.body, include_for_targets=False)
        body_bound = self._body_bound_names(stmt.body, include_for_targets=True)
        # iterate pre_definite (insertion order) for deterministic phi order;
        # carried includes nested for-targets (they bind in the enclosing
        # scope, so rebinding them is a cross-iteration dependency)
        carried = {v: pre_definite[v] for v in pre_definite if v in body_bound}
        if loop_var in body_assigned:
            raise self.error(stmt, f"cannot assign the loop variable {loop_var!r} inside its loop")

        phi_kinds: dict[str, ir.Kind | None] = {}
        # a pre-bound loop variable keeps its value on zero-trip loops:
        # emit a phi (the for-statement overwrites it on real iterations)
        if loop_var in pre_definite:
            self.bindings.append(ir.Binding(loop_name, ir.Ref(pre_definite[loop_var])))
            self.kinds.setdefault_kind(loop_name, self.kinds.kind(pre_definite[loop_var]))

        # phis for loop-carried variables, emitted just before the loop
        for var, pre_name in carried.items():
            loop_carried_name = self.ssa.bind(var)
            carried[var] = loop_carried_name
            self.bindings.append(ir.Binding(loop_carried_name, ir.Ref(pre_name)))
            self.definite[var] = loop_carried_name
            phi_kinds[loop_carried_name] = self.kinds.kind(pre_name)
            self.kinds.set_kind(loop_carried_name, self.kinds.kind(pre_name))
            # the phi feeds the pre-loop value on zero-trip loops: a raw
            # scalar stays a raw scalar, so track that for call sites
            if self.kinds.is_scalar(pre_name):
                self.kinds.mark_scalar(loop_carried_name)
            if self.kinds.kind(pre_name) is None or self.kinds.is_maybe_bool(pre_name):
                # Unknown entry values can be boolean arrays even when
                # every body assignment has the same unknown static kind.
                self.kinds.mark_maybe_bool(loop_carried_name)

        phi_end = len(self.bindings)

        def union_labels(
            frame: dict[str, set[ir.Kind | None]],
        ) -> tuple[dict[str, ir.Kind | None], list[str]]:
            """Per carried name: the final kind and the mixed names.

            The final kind is the union of the phi kind and all
            body-assignment kinds; a name is mixed when its kinds MIX
            across iterations.
            """
            labels: dict[str, ir.Kind | None] = {}
            mixed: list[str] = []
            for loop_carried_name in carried.values():
                kinds = set[ir.Kind | None]((phi_kinds.get(loop_carried_name),)) | frame.get(
                    loop_carried_name, set()
                )
                unique: set[ir.Kind] = {k for k in kinds if k is not None}
                if (None in kinds and unique) or len(unique) > 1:
                    # mixed (or unknown-mixed) kinds across iterations:
                    # label 'bool' so exact-arithmetic intify kicks in at
                    # uses inside the body (the intify casts to the sibling
                    # operand's runtime dtype, so no truncation is possible)
                    labels[loop_carried_name] = "bool"
                    mixed.append(loop_carried_name)
                else:
                    labels[loop_carried_name] = unique.pop() if unique else None
            return labels, mixed

        # The body is lowered repeatedly until per-name kinds reach a fixed
        # point. A carried variable's kind can mix across iterations (int
        # entry, bool body assignment, ...), and the mixing can propagate
        # through assignment chains (a = b; b = c; c = x > 0), so one pass
        # cannot see every name that needs the intify conversion. Labels
        # only ever WIDEN (int/float -> bool), so at most len(carried)
        # widenings can happen; bound the passes at len(carried) + 2.
        labels: dict[str, ir.Kind | None] = {}
        prev_labels: dict[str, ir.Kind | None] | None = None
        max_passes = len(carried) + 2
        for attempt in range(max_passes):
            pre_body = (
                dict(self.definite),
                set(self.maybe),
                list(self.deferred),
                self.kinds.snapshot_facts(),
            )
            self.kinds.push_carried_frame()
            self.loop_depth += 1
            self.loop_vars.add(loop_var)
            self.active_carried.append((dict(carried), self.branch_depth))
            self.definite[loop_var] = loop_name
            self.kinds.setdefault_kind(loop_name, "int")  # loop vars are Python ints
            self.kinds.mark_scalar(loop_name)  # raw Python ints per iteration
            self.maybe.discard(loop_var)
            try:
                result = self.lower_stmts(stmt.body, loop_body=True)
            finally:
                self.loop_depth -= 1
                self.loop_vars.discard(loop_var)
                self.active_carried.pop()
                frame = self.kinds.pop_carried_frame()
            if result is not None:  # pragma: no cover - returns rejected in bodies
                raise self.error(stmt, "returns inside loop bodies are not supported")
            labels, mixed = union_labels(frame)
            if labels == prev_labels or not mixed or attempt == max_passes - 1:
                break
            # restore the pre-body state and re-lower with the widened
            # labels pre-applied (kind + literal facts only; scalar/bool
            # flags persist across passes)
            prev_labels = labels
            (definite, maybe, deferred, facts) = pre_body
            del self.bindings[phi_end:]
            self.definite, self.maybe, self.deferred = definite, maybe, deferred
            self.kinds.restore_facts(facts)
            self.kinds.update_kinds(labels)
        for loop_carried_name, kind in labels.items():
            self.kinds.set_kind(loop_carried_name, kind)
        # literal facts are invalid across the loop boundary: the phi feeds
        # pre-loop values on zero-trip loops, and body assignments feed
        # later iterations — neither is the recorded literal
        for loop_carried_name in carried.values():
            self.kinds.drop_literal(loop_carried_name)
        body_stmts = tuple(self.bindings[phi_end:])
        del self.bindings[phi_end:]
        self.bindings.append(ir.Loop(loop_name, start, stop, step, body_stmts))

        # loop-local variables (assigned in body, not bound before) keep their
        # body names: zero-trip loops raise NameError exactly like Python.
        # The loop variable stays bound after the loop (last iteration value).

    def _sync_loop_carried(self) -> None:
        """Copy rebound values back to their active loop-carried names."""
        for carried, _entry_depth in reversed(self.active_carried):
            for var, loop_name in carried.items():
                current = self.definite.get(var)
                if current is not None and current != loop_name:
                    self.bindings.append(ir.Binding(loop_name, ir.Ref(current)))
                    self.definite[var] = loop_name
                    self.kinds.set_kind(loop_name, self.kinds.kind(current))
                    self.kinds.drop_literal(loop_name)
