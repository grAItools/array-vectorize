# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def docstring_google_vec(x, y):
    '''
    (array-vectorized) Add two values, scaled.

    Long description exercising the summary/body split.

    Args:
        x: first value
        y: second value

    Returns:
        the scaled sum

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def docstring_google(x, y):
                """Add two values, scaled.

                Long description exercising the summary/body split.

                Args:
                    x: first value
                    y: second value

                Returns:
                    the scaled sum
                """
                return (x + y) * SCALE
    '''
    xp = array_namespace(x, y)
    return (x + y) * 3.0
