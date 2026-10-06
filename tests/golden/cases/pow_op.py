from array_api_compat import array_namespace

def pow_op_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def pow_op(x):
                return x**2 + 2.0**x
    """
    xp = array_namespace(x)
    return xp.pow(x, 2) + xp.pow(2.0, x)
