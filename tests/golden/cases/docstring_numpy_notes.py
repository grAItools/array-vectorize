# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

from array_api_compat import array_namespace

def docstring_numpy_notes_vec(x):
    '''
    (array-vectorized) Double the input.

    Notes
    -----
    Pre-existing note.

    Vectorized by array-vectorize from this scalar original::

        def docstring_numpy_notes(x):
            """Double the input.

            Notes
            -----
            Pre-existing note.
            """
            return x * 2.0
    '''
    xp = array_namespace(x)
    return x * 2.0
