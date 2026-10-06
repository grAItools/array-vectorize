from array_api_compat import array_namespace

def compare_ops_vec(x, y):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def compare_ops(x, y):
                return (x == y) + (x != y) + (x < y) + (x <= y) + (x > y) + (x >= y)
    """
    xp = array_namespace(x, y)
    return _vec_arith(xp, 1, _vec_arith(xp, 1, _vec_arith(xp, 1, _vec_arith(xp, 1, _vec_arith(xp, 1, x == y, x != y), x < y), x <= y), x > y), x >= y)
