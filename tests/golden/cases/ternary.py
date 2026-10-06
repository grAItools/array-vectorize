from array_api_compat import array_namespace

def ternary_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def ternary(x):
                return x * 2 if x > 0 else x / 2
    """
    xp = array_namespace(x)
    return xp.where(x > 0, x * 2, x / 2)
