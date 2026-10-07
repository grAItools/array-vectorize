# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

def add_vec(x, y, *, _namespace=None):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def add(x, y):
                return x + y
    """
    xp = _namespace
    return x + y
