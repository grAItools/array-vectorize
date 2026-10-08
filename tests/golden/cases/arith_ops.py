# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def arith_ops_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def arith_ops(x, y):
                return (x + y - x * y) / (x * x + y * y)
    """
    xp = array_namespace(x, y)
    return _vec_arith(xp, 4, x + y - x * y, x * x + y * y)
