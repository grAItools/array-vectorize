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
