# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def minmax_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def minmax(x, y):
                return min(x, y) + max(x, y, 3.0)
    """
    xp = array_namespace(x, y)
    t_1 = xp.asarray(x)
    t_2 = xp.asarray(y)
    return _vec_arith(xp, 1, _vec_minmax(xp, True, t_1, t_2), _vec_minmax(xp, False, t_1, t_2, 3.0))
