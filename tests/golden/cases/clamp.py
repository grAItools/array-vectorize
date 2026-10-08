# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def clamp_vec(x, lo=0.0, hi=1.0):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def clamp(x, lo=0.0, hi=1.0):
                if x < lo:
                    return lo
                if x > hi:
                    return hi
                return x
    """
    xp = array_namespace(x, lo, hi)
    return xp.where(x < lo, lo, xp.where(x > hi, hi, x))
