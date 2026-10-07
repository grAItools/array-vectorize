# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def relu_with_else_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def relu_with_else(x):
                if x > 0:  # noqa: SIM108
                    r = x
                else:
                    r = 0.0
                return r
    """
    xp = array_namespace(x)
    r_1 = x
    r_2 = 0.0
    r = xp.where(x > 0, r_1, r_2)
    return r
