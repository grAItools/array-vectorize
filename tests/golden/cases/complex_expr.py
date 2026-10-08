# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def complex_expr_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def complex_expr(x, y):
                t = math.exp(-(x * x + y * y))
                return t / (1.0 + t)
    """
    xp = array_namespace(x, y)
    t = xp.exp(xp.astype(xp.asarray(-(x * x + y * y)), xp.float64))
    return _vec_arith(xp, 4, t, 1.0 + t)
