from array_api_compat import array_namespace

def psi_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def psi(x):
                if x < 0:
                    return 0.0
                return x * math.exp(-x)
    """
    xp = array_namespace(x)
    return xp.where(x < 0, 0.0, x * xp.exp(xp.astype(xp.asarray(-x), xp.float64)))
