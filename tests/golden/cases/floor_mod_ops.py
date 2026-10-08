# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def floor_mod_ops_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def floor_mod_ops(x):
                return (x // 2) + (x % 3)
    """
    xp = array_namespace(x)
    return xp.floor_divide(x, 2) + xp.remainder(x, 3)
