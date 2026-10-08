# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def helper_outer_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def helper_outer(x):
                return helper_inner(x) + helper_inner(x * 2.0)
    """
    xp = array_namespace(x)
    return helper_inner_vec(x) + helper_inner_vec(x * 2.0)
