# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def loop_start_stop_step_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def loop_start_stop_step(x):
                s = 1.0
                for i in range(2, 8, 3):
                    s = s * i
                return s + x
    """
    xp = array_namespace(x)
    s = 1.0
    s_1 = s
    for i in range(2, 8, 3):
        s_1 = s_1 * i
    return s_1 + x
