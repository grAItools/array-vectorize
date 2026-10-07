# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def compare_chain_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def compare_chain(x):
                return 0 < x < 1
    """
    xp = array_namespace(x)
    return xp.logical_and(0 < x, x < 1)
