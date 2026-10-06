from array_api_compat import array_namespace

def docstring_existing_notes_vec(x):
    '''
    (array-vectorized) Clamp nonnegative values.

    Notes:
        Pre-existing note.

        Vectorized by array-vectorize from this scalar original::

            def docstring_existing_notes(x):
                """Clamp nonnegative values.

                Notes:
                    Pre-existing note.
                """
                if x < 0.0:
                    return 0.0
                return x
    '''
    xp = array_namespace(x)
    return xp.where(x < 0.0, 0.0, x)
