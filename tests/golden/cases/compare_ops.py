from array_api_compat import array_namespace

def compare_ops_vec(x, y):
    """def compare_ops(x, y):
    return (x == y) + (x != y) + (x < y) + (x <= y) + (x > y) + (x >= y)"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    t_1 = xp.asarray(x == y)
    t_2 = xp.asarray(x != y)
    t_3 = xp.asarray(x < y)
    t_4 = xp.asarray(x <= y)
    t_5 = xp.asarray(x > y)
    t_6 = xp.astype(t_1, _vec_arith_dtype(xp, 1, x == y, x != y))
    t_7 = xp.astype(t_2, _vec_arith_dtype(xp, 1, x == y, x != y))
    t_8 = xp.asarray(t_6 + t_7)
    t_9 = xp.astype(t_3, _vec_arith_dtype(xp, 1, t_6 + t_7, x < y))
    t_10 = xp.astype(t_8, _vec_arith_dtype(xp, 1, t_6 + t_7, x < y))
    t_11 = xp.asarray(t_10 + t_9)
    t_12 = xp.astype(t_4, _vec_arith_dtype(xp, 1, t_10 + t_9, x <= y))
    t_13 = xp.astype(t_11, _vec_arith_dtype(xp, 1, t_10 + t_9, x <= y))
    t_14 = xp.asarray(t_13 + t_12)
    t_15 = xp.astype(t_5, _vec_arith_dtype(xp, 1, t_13 + t_12, x > y))
    t_16 = xp.astype(t_14, _vec_arith_dtype(xp, 1, t_13 + t_12, x > y))
    return xp.astype(xp.asarray(t_16 + t_15), _vec_arith_dtype(xp, 1, t_16 + t_15, x >= y)) + xp.astype(xp.asarray(x >= y), _vec_arith_dtype(xp, 1, t_16 + t_15, x >= y))
