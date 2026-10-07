# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def boolop_numeric_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def boolop_numeric(x, y):
                return x and y
    """
    xp = array_namespace(x, y)
    return xp.where(x != 0, y, x)
