from array_api_compat import array_namespace

def bitwise_ops_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def bitwise_ops(x, y):
                return (x & y) | (x ^ 3) << 1 >> 2
    """
    xp = array_namespace(x, y)
    return x & y | (x ^ 3) << 1 >> 2
