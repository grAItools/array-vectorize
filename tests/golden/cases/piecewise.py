from array_api_compat import array_namespace

def piecewise_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def piecewise(x):
                if x < -1:
                    return -1.0
                elif x > 1:
                    return 1.0
                return x
    """
    xp = array_namespace(x)
    return xp.where(x < -1, -1.0, xp.where(x > 1, 1.0, x))
