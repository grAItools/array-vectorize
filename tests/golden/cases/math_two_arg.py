# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def math_two_arg_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def math_two_arg(x, y):
                return math.atan2(x, y) + math.hypot(x, y) + math.copysign(x, y)
    """
    xp = array_namespace(x, y)
    t_1 = xp.asarray(x)
    t_2 = xp.asarray(y)
    t_3 = xp.astype(t_1, xp.float64)
    t_4 = xp.astype(t_2, xp.float64)
    return xp.atan2(t_3, t_4) + xp.hypot(t_3, t_4) + xp.copysign(t_3, t_4)
