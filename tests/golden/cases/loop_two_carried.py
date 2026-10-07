# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def loop_two_carried_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def loop_two_carried(x):
                a = 0.0
                b = 1.0
                for i in range(3):  # noqa: B007
                    a = a + x
                    b = b * 2.0
                return a + b
    """
    xp = array_namespace(x)
    a = 0.0
    b = 1.0
    a_1 = a
    b_1 = b
    for i in range(0, 3):
        a_1 = a_1 + x
        b_1 = b_1 * 2.0
    return a_1 + b_1
