# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def cse_opportunity_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def cse_opportunity(x):
                return math.sqrt(x) * math.sqrt(x) + math.sqrt(x)
    """
    xp = array_namespace(x)
    t_1 = xp.asarray(x)
    t_2 = xp.astype(t_1, xp.float64)
    t_3 = xp.sqrt(t_2)
    return t_3 * t_3 + t_3
