# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""User-helper lowering: vectorizing and calling other scalar functions.

Calls to functions in the enclosing scope (``info.user_funcs``) are
vectorized through the pipeline's helper callback and emitted as calls
to the vectorized version, with collision-free allocated names and
memoization per callee.
"""

from __future__ import annotations

import ast

from array_vectorize import ir
from array_vectorize.lower import loops

__all__ = ["_HelperLowerer"]


class _HelperLowerer(loops._LoopLowerer):
    def _call_helper(self, node: ast.Call, name: str, args: list[ir.Node]) -> ir.Node:
        """Vectorize and call another scalar function (design D7)."""
        if self.helper_vectorizer is None:
            raise self.error(
                node,
                f"call to {name!r} requires the vectorize pipeline "
                "(helper functions are only supported through vectorize())",
            )
        callee = self.info.user_funcs[name]
        if callee in self._helper_names:
            return ir.FuncCall(self._helper_names[callee], tuple(args))
        base = ir.generated_name(callee.__name__)
        while not self.ssa.is_free(base):
            base += "_"
        self.ssa.reserve(base)
        vec = self.helper_vectorizer(callee)
        self.helpers.append((base, vec))
        self._helper_names[callee] = base
        return ir.FuncCall(base, tuple(args))
