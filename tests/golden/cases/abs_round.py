from array_api_compat import array_namespace

def abs_round_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def abs_round(x):
                return abs(x) + round(x)
    """
    xp = array_namespace(x)
    t_1 = xp.asarray(x)
    return xp.abs(t_1) + xp.round(t_1)
