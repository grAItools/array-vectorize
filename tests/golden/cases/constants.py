# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def constants_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def constants(x):
                return x * math.pi + math.e + math.inf - math.nan
    """
    xp = array_namespace(x)
    return x * 3.141592653589793 + 2.718281828459045 + xp.inf - xp.nan
