"""Expression lowering: AST expressions to IR nodes.

Covers binops (with the exactness fixes that route boolean-adjacent
arithmetic through the runtime helper), comparisons, boolean operators,
conditional expressions, and calls — math functions, polymorphic
builtins (abs/min/max), and dtype casts — plus the numeric-kind and
maybe-bool inference that decides where those exactness mitigations
fire.
"""

from __future__ import annotations

import ast

from array_vectorize import ir
from array_vectorize.frontend import tables
from array_vectorize.lower import sanitize
from array_vectorize.optimize import constfold
from array_vectorize.runtime import registry

__all__ = ["_ExpressionLowerer"]

_BINOPS: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.BitAnd: "and",
    ast.BitOr: "or",
    ast.BitXor: "xor",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
}
_CMPOPS: dict[type[ast.cmpop], str] = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
_UNARY: dict[type[ast.unaryop], str] = {
    ast.USub: "neg",
    ast.UAdd: "pos",
    ast.Invert: "invert",
    ast.Not: "not",
}


class _ExpressionLowerer(sanitize._Sanitizer):
    # --------------------------------------------------------- expressions

    def lower_expr(self, node: ast.expr) -> ir.Node:
        match node:
            case ast.Constant():
                return self._literal(node.value)
            case ast.Name():
                return self.load_name(node)
            case ast.BinOp():
                return self._lower_binop(
                    node.op, self.lower_expr(node.left), self.lower_expr(node.right)
                )
            case ast.UnaryOp():
                operand = self.lower_expr(node.operand)
                if isinstance(node.op, ast.USub) and (
                    self._numeric_kind(operand) == "bool" or self._maybe_bool_result(operand)
                ):
                    # -True is -1 in Python; numpy cannot negate bool arrays.
                    # The whole negation goes through the runtime helper (it
                    # computes the negation — no wrapping UnaryOp on top)
                    if isinstance(operand, ir.Literal):
                        return ir.Literal(-int(operand.value), "int")
                    return ir.FuncCall(
                        self.arith_name,
                        (
                            ir.Ref(self.ns_var),
                            ir.Literal(int(registry.ArithOp.NEG), "int"),
                            operand,
                        ),
                    )
                return ir.UnaryOp(_UNARY[type(node.op)], operand)
            case ast.Compare():
                return self._lower_compare(node)
            case ast.BoolOp():
                return self._lower_boolop(node)
            case ast.IfExp():
                return ir.Where(
                    self._coerce_bool(self.lower_expr(node.test)),
                    self.lower_expr(node.body),
                    self.lower_expr(node.orelse),
                )
            case ast.Call():
                return self._lower_call(node)
            case ast.Attribute():
                return self._lower_math_const(node)
            case _:
                # ast.expr is an open set (new syntax per Python release);
                # the validator rejects these first, lowering never sees them
                raise self.error(node, f"{type(node).__name__} expressions are not supported")

    def _lower_compare(self, node: ast.Compare) -> ir.Node:
        parts: list[ir.Node] = []
        operands = [node.left, *node.comparators]
        for left, op, right in zip(operands, node.ops, operands[1:], strict=False):
            parts.append(
                ir.Compare(_CMPOPS[type(op)], self.lower_expr(left), self.lower_expr(right))
            )
        if len(parts) == 1:
            return parts[0]
        return ir.Logical("and", tuple(parts))

    def _lower_boolop(self, node: ast.BoolOp) -> ir.Node:
        op = "and" if isinstance(node.op, ast.And) else "or"
        values = [self.lower_expr(v) for v in node.values]
        if all(ir.is_bool(v) for v in values):
            return ir.Logical(op, tuple(values))
        # exact value-select lowering (design D2)
        partial = values[0]
        last = len(values) - 1
        for i, value in enumerate(values[1:], 1):
            if op == "and":
                merged: ir.Node = ir.Where(self._coerce_bool(partial), value, partial)
            else:
                merged = ir.Where(self._coerce_bool(partial), partial, value)
            if i < last:
                # bind intermediates to temps; only the final value stays inline
                temp = self.ssa.fresh_temp("v")
                self.bindings.append(ir.Binding(temp, merged))
                partial = ir.Ref(temp)
            else:
                partial = merged
        return partial

    def _lower_math_const(self, node: ast.Attribute) -> ir.Node:
        return self._literal(tables.MATH_CONSTS[node.attr])

    def _lower_call(self, node: ast.Call) -> ir.Node:
        args = [self.lower_expr(a) for a in node.args]
        func = node.func
        if isinstance(func, ast.Attribute):
            return self._call_math(func, func.attr, args, node)
        assert isinstance(func, ast.Name)
        name = func.id
        info = self.info
        if name in self.definite or name in self.maybe or name in self.assigned_names:
            raise self.error(node, f"calling variable {name!r} is not supported")
        if name in info.math_funcs:
            return self._call_math(func, info.math_funcs[name], args, node)
        if name in info.user_funcs:
            return self._call_helper(node, name, args)
        if name in info.math_modules:
            raise self.error(node, "the math module cannot be called")
        if name == "round" and len(args) != 1:
            raise self.error(
                node,
                "round(x, n) is not supported: the decimals kwarg is not in the Array API standard",
            )
        if name in tables.BUILTIN_UNARY:
            self._check_arity(node, name, args, 1, 1)
            sanitized = tuple(self._sanitize_xp_arg(a, force_float=False) for a in args)
            return ir.Call(tables.BUILTIN_UNARY[name], sanitized)
        if name in tables.BUILTIN_FOLDS:
            if len(args) < 2:
                raise self.error(node, f"{name}() needs at least 2 arguments in vectorized code")
            if all(self._literal_value(a) is not None for a in args):
                # all-literal call: substitute the actual literals so
                # const-folding fires (Ref nodes would block it and reach
                # codegen as raw Python scalars)
                clean: list[ir.Node] = [
                    lit for a in args if (lit := self._literal_value(a)) is not None
                ]
            else:
                # one runtime helper call with exact selection semantics:
                # identical-dtype fast paths, never-winning-bound clamps,
                # proven integer result ranges (max(uint64, signed) fits
                # uint64; min(uint64, signed) fits int64), Python float
                # promotion. Literal args pass as raw values; array args
                # pass through the soft sanitizer
                is_min = tables.BUILTIN_FOLDS[name] == "minimum"
                helper_args: list[ir.Node] = [ir.Ref(self.ns_var), ir.Literal(is_min, "bool")]
                for a in args:
                    lit = self._literal_value(a)
                    if lit is None:
                        helper_args.append(self._sanitize_xp_arg(a, force_float=False))
                    else:
                        helper_args.append(lit)
                return ir.FuncCall(self.minmax_name, tuple(helper_args))
            folded: ir.Node = ir.Call(tables.BUILTIN_FOLDS[name], (clean[0], clean[1]))
            for arg in clean[2:]:
                folded = ir.Call(tables.BUILTIN_FOLDS[name], (folded, arg))
            return folded
        if name in tables.BUILTIN_CASTS:
            self._check_arity(node, name, args, 1, 1)
            return self._cast_call(args[0], tables.BUILTIN_CASTS[name])
        raise self.error(node, f"unsupported function {name!r}")

    def _cast_call(self, arg: ir.Node, dtype: str) -> ir.Node:
        """Lower an int()/float()/bool()/math.trunc-style cast.

        Literal args fold exactly, raw loop-variable ints wrap in asarray,
        and arrays pass through (params are preamble-normalized).
        """
        lit = self._literal_value(arg)
        if lit is not None:
            try:
                if dtype == "bool":
                    return ir.Literal(bool(lit.value), "bool")
                if dtype == "int64":
                    return ir.Literal(int(lit.value), "int")
                return ir.Literal(float(lit.value), "float")
            except (ValueError, OverflowError):
                pass  # inf/nan literals: leave the cast to runtime astype
        return ir.Call("astype", (self._sanitize_xp_arg(arg, force_float=False), ir.DType(dtype)))

    def _call_math(self, func: ast.AST, attr: str, args: list[ir.Node], node: ast.AST) -> ir.Node:
        if attr in tables.MATH_SPECIAL:
            self._check_arity(node, attr, args, 1, 1)
            return self._cast_call(args[0], tables.MATH_SPECIAL[attr])
        if attr in tables.MATH_FUNCS:
            xp_name, lo, hi = tables.MATH_FUNCS[attr]
            self._check_arity(node, attr, args, lo, hi)
            # Array API elementwise math requires floating-point array
            # inputs; sanitize literal/int/bool/possibly-scalar args
            sanitized = tuple(self._sanitize_xp_arg(a, force_float=True) for a in args)
            return ir.Call(xp_name, sanitized)
        raise self.error(func, f"math.{attr} cannot be called")

    @staticmethod
    def _is_negative_int(node: ir.Node) -> bool:
        """True for literal negative ints (including UnaryOp(neg, lit))."""
        if isinstance(node, ir.Literal) and node.kind == "int":
            return node.value < 0
        return (
            isinstance(node, ir.UnaryOp)
            and node.op == "neg"
            and isinstance(node.operand, ir.Literal)
            and node.operand.kind == "int"
        )

    def _numeric_kind(self, node: ir.Node) -> ir.Kind | None:
        """Best-effort numeric kind of ``node``.

        Literals, casts, bools, arithmetic, and Refs whose binding kind
        was recorded are provable; anything else reads as ``None``.
        """
        match node:
            case ir.Literal():
                return node.kind
            case ir.Ref():
                return self.kinds.kind(node.name)
            case ir.Logical():
                return "bool"
            case ir.Where():
                kt, ke = self._numeric_kind(node.then), self._numeric_kind(node.other)
                return kt if kt == ke else None
            case _ if ir.is_bool(node):
                # provably-boolean nodes (Compare, not, isnan/isinf/isfinite)
                return "bool"
            case ir.BinOp():
                # scalar promotion rules: int/bool + float -> float; arithmetic
                # over ints/bools stays int; true division is ALWAYS float in
                # Python (int / int -> float), whatever the operand kinds
                lk, rk = self._numeric_kind(node.left), self._numeric_kind(node.right)
                if node.op == "div":
                    return "float"
                if node.op in ("and", "or", "xor"):
                    # bitwise: Python bool & bool is bool (saturating array bool
                    # arithmetic would otherwise lose exactness downstream)
                    if lk == "bool" and rk == "bool":
                        return "bool"
                    if lk in ("int", "bool") and rk in ("int", "bool"):
                        return "int"
                    return None
                if "float" in (lk, rk):
                    return "float"
                if lk in ("int", "bool") and rk in ("int", "bool"):
                    return "int"
                return None
            case ir.UnaryOp() if node.op in ("neg", "pos"):
                return self._numeric_kind(node.operand)
            case ir.Call() if node.fn == "asarray" and len(node.args) == 1:
                # asarray preserves the wrapped value's kind
                return self._numeric_kind(node.args[0])
            case ir.FuncCall() if node.fn == self.minmax_name:
                # min/max result kind: bool when every argument is bool, float
                # when any argument is float, int for int-ish mixes (the helper
                # normalizes bool arrays to int8, so arithmetic downstream must
                # intify)
                kinds = [self._numeric_kind(a) for a in node.args[2:]]
                if any(k is None for k in kinds):
                    return None
                if "float" in kinds:
                    return "float"
                if all(k == "bool" for k in kinds):
                    return "bool"
                return "int"
            case ir.FuncCall() if node.fn == self.arith_name:
                # arithmetic helper result kind: floats stay float, true
                # division is float, known int/bool operands give int (bools
                # are converted to a numeric dtype inside the helper) — so
                # downstream mitigations (e.g. the negative-exponent float
                # cast) keep firing for provably-integer results
                op_lit = node.args[1] if len(node.args) > 1 else None
                op_id = op_lit.value if isinstance(op_lit, ir.Literal) else None
                if op_id == int(registry.ArithOp.DIV):
                    return "float"
                kinds = [self._numeric_kind(a) for a in node.args[2:]]
                if any(k is None for k in kinds):
                    return None
                if "float" in kinds:
                    return "float"
                return "int"
            case ir.Call() if (
                node.fn == "astype" and len(node.args) == 2 and isinstance(node.args[1], ir.DType)
            ):
                name = node.args[1].name
                return "bool" if name == "bool" else "int" if name == "int64" else "float"
            case _:
                # no provable kind (unknown ops, untracked casts, ...)
                return None

    def _maybe_bool_result(self, node: ir.Node) -> bool:
        """True for min/max results whose static kind is unknown.

        Their runtime dtype may be boolean (parameter operands), so
        arithmetic must use the runtime-polymorphic intify — the
        _vec_arith_dtype cast is a no-op for numeric dtypes and int64
        for booleans.
        """
        if isinstance(node, ir.FuncCall) and node.fn == self.minmax_name:
            return self._numeric_kind(node) is None
        if isinstance(node, ir.Ref):
            return self.kinds.is_maybe_bool(node.name)
        if isinstance(node, ir.Where):
            # branch merge of maybe-bool values is maybe-bool
            return self._maybe_bool_result(node.then) or self._maybe_bool_result(node.other)
        if isinstance(node, ir.BinOp) and node.op in ("and", "or", "xor"):
            # bitwise expressions preserve boolean-ness (arithmetic
            # results are already intified and numeric)
            return self._maybe_bool_result(node.left) or self._maybe_bool_result(node.right)
        return False

    def _lower_binop(self, op: ast.operator, left: ir.Node, right: ir.Node) -> ir.Node:
        """Build a BinOp with the exactness fixes shared by `a op b` and `a op= b`."""
        if isinstance(op, ast.FloorDiv | ast.Mod):
            literal_left, literal_right = self._literal_value(left), self._literal_value(right)
            if literal_left is not None and literal_right is not None:
                # Keep foldable constants usable as loop bounds, while
                # zero divisors and overflow retain runtime dispatch.
                folded = constfold._fold_binop(_BINOPS[type(op)], literal_left, literal_right)
                if folded is not None:
                    return folded
        # Python bool arithmetic is integer (True + True == 2), but array
        # backends either saturate (bool + bool == True) or reject bools in
        # arithmetic outright (strict backends). Runtime dispatch chooses
        # a numeric dtype without rounding integer operands, including
        # loop-carried values whose kind changes across iterations.
        if isinstance(op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.FloorDiv | ast.Mod) and (
            "bool" in (self._numeric_kind(left), self._numeric_kind(right))
            or self._maybe_bool_result(left)
            or self._maybe_bool_result(right)
            # True division requires floating operands even when input
            # kinds are unknown. Negative integer literals cannot be
            # promoted into uint64 by a bare backend operator.
            or (
                isinstance(op, ast.Div)
                and (self._numeric_kind(left) != "float" or self._numeric_kind(right) != "float")
            )
            or (
                isinstance(op, ast.FloorDiv | ast.Mod)
                and (
                    self._is_negative_int(self._literal_value(left) or left)
                    or self._is_negative_int(self._literal_value(right) or right)
                )
            )
        ):
            # the whole operation goes through the runtime _vec_arith
            # helper: bools need a numeric representation, strict
            # backends require single-dtype or float operands, and the
            # uint64 integer paths are exact per lane (modular for
            # add/sub/mul; int64 magnitudes for negative divisors)
            if isinstance(op, ast.FloorDiv | ast.Mod):
                left = self._literal_value(left) or left
                right = self._literal_value(right) or right
            ir_op = _BINOPS[type(op)]
            return ir.FuncCall(
                self.arith_name,
                (
                    ir.Ref(self.ns_var),
                    ir.Literal(int(registry.ArithOp[ir_op.upper()]), "int"),
                    left,
                    right,
                ),
            )
        if isinstance(op, ast.Pow) and self._is_negative_int(right):
            # Python promotes int ** negative-int to float; NumPy raises.
            # Cast provably-int bases so the common literal case matches.
            left = self._as_float_arg(left)
        return ir.BinOp(_BINOPS[type(op)], left, right)

    def _check_arity(self, node: ast.AST, name: str, args: list[ir.Node], lo: int, hi: int) -> None:
        if not lo <= len(args) <= hi:
            expected = str(lo) if lo == hi else f"{lo}-{hi}"
            raise self.error(node, f"{name}() takes {expected} argument(s), got {len(args)}")
