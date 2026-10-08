# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def ternary_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def ternary(x):
                return x * 2 if x > 0 else x / 2
    """
    xp = array_namespace(x)
    return xp.where(x > 0, x * 2, _vec_arith(xp, 4, x, 2))
