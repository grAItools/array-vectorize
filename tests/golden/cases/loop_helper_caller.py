# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def loop_helper_caller_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def loop_helper_caller(x):
                s = 0.0
                for i in range(3):  # noqa: B007
                    s = s + helper_inner(x)
                return s
    """
    xp = array_namespace(x)
    s = 0.0
    s_1 = s
    for i in range(0, 3):
        s_1 = s_1 + helper_inner_vec(x)
    return s_1
