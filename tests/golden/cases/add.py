from array_api_compat import array_namespace

def add_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def add(x, y):
                return x + y
    """
    xp = array_namespace(x, y)
    return x + y
