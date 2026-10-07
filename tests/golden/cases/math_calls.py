# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def math_calls_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def math_calls(x):
                return math.sqrt(x) + math.exp(x) + math.log1p(x) + math.sin(x)
    """
    xp = array_namespace(x)
    t_1 = xp.asarray(x)
    t_2 = xp.astype(t_1, xp.float64)
    return xp.sqrt(t_2) + xp.exp(t_2) + xp.log1p(t_2) + xp.sin(t_2)
