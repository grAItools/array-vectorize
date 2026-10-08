# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def ssa_rebind_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def ssa_rebind(x):
                y = x + 1
                y = y * 2
                y += 3
                return y
    """
    xp = array_namespace(x)
    y = x + 1
    y_1 = y * 2
    y_2 = y_1 + 3
    return y_2
