from array_api_compat import array_namespace

def unary_ops_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def unary_ops(x):
                return -x + +x - ~x
    """
    xp = array_namespace(x)
    return -x + +x - ~x
