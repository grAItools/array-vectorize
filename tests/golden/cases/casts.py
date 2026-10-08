# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def casts_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def casts(x):
                return int(x) + float(x) + bool(x) + math.trunc(x)
    """
    xp = array_namespace(x)
    t_1 = xp.asarray(x)
    t_2 = xp.astype(t_1, xp.int64)
    return _vec_arith(xp, 1, t_2 + xp.astype(t_1, xp.float64), xp.astype(t_1, xp.bool)) + t_2
