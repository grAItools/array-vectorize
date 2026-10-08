# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Argument sanitization for Array API call sites.

Operators promote raw Python scalars fine, but ``xp.*`` function
arguments must be arrays (strict backends reject plain scalars) and
elementwise math requires floating-point arrays on every backend. This
layer decides, per argument, whether a value needs wrapping in
``asarray`` or a float64 cast, and resolves the literal behind an
argument — directly, negated, or via a literal binding.
"""

from __future__ import annotations

from array_vectorize import ir
from array_vectorize.ir import walk as walk_mod
from array_vectorize.lower import env

__all__ = ["_Sanitizer"]


class _Sanitizer(env._LowererBase):
    def _literal_value(self, arg: ir.Node) -> ir.Literal | None:
        """The literal behind an arg, or ``None``.

        Found directly, as a negated literal (``-1`` lowers to
        UnaryOp(neg, 1)), or via a literal binding.
        """
        if isinstance(arg, ir.Literal):
            return arg
        if (
            isinstance(arg, ir.UnaryOp)
            and arg.op == "neg"
            and isinstance(arg.operand, ir.Literal)
            and arg.operand.kind in ("int", "float")
        ):
            return ir.Literal(-arg.operand.value, arg.operand.kind)
        if isinstance(arg, ir.Ref):
            return self.kinds.literal(arg.name)
        return None

    def _possibly_scalar(self, node: ir.Node) -> bool:
        """True when the node's runtime value may be a raw Python scalar.

        It is a literal, a possibly-scalar name (parameter default, loop
        variable, literal binding), or an expression computed from one.
        Operators promote such scalars fine; xp.* function arguments do
        not (strict backends reject plain scalars).
        """

        def walk(n: ir.Node) -> bool:
            if isinstance(n, ir.Literal):
                return True
            if isinstance(n, ir.Ref):
                return self.kinds.is_scalar(n.name)
            if isinstance(n, ir.DType):
                return False
            return any(walk(c) for c in walk_mod.children(n))

        return walk(node)

    def _sanitize_xp_arg(self, arg: ir.Node, *, force_float: bool) -> ir.Node:
        """Make an argument safe for xp.* calls.

        Operators promote raw Python scalars fine, but xp.* function
        arguments must be arrays (strict backends reject plain scalars) —
        and elementwise math requires floating-point arrays on every
        backend.

        ``force_float`` (math functions): scalar semantics convert to
        double, so literals become float literals (const-folded away) and
        provably int/bool or possibly-scalar args are cast to float64.

        Otherwise (polymorphic builtins such as abs/min/max and dtype
        casts): keep values exact — literals and possibly-scalar args are
        wrapped in asarray, with no dtype conversion.
        """
        lit = self._literal_value(arg)
        if lit is not None:
            if force_float:
                return ir.Literal(float(lit.value), "float")
            return ir.Call("asarray", (lit,))
        kind = self._numeric_kind(arg)
        if kind in ("int", "bool") or self._possibly_scalar(arg):
            if force_float:
                return ir.Call("astype", (ir.Call("asarray", (arg,)), ir.DType("float64")))
            return ir.Call("asarray", (arg,))
        return arg

    def _as_float_arg(self, arg: ir.Node) -> ir.Node:
        kind = self._numeric_kind(arg)
        if kind in ("int", "bool"):
            if isinstance(arg, ir.Literal):
                return ir.Literal(float(arg.value), "float")
            return ir.Call("astype", (ir.Call("asarray", (arg,)), ir.DType("float64")))
        return arg
