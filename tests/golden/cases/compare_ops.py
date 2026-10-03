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
    t_6 = xp.astype(t_1, _vec_common_dtype(xp, x != y))
    t_7 = xp.astype(t_2, _vec_common_dtype(xp, t_6))
    t_8 = xp.astype(t_3, _vec_common_dtype(xp, t_6 + t_7))
    t_9 = xp.astype(t_4, _vec_common_dtype(xp, t_6 + t_7 + t_8))
    t_10 = xp.astype(t_5, _vec_common_dtype(xp, t_6 + t_7 + t_8 + t_9))
    return t_6 + t_7 + t_8 + t_9 + t_10 + xp.astype(xp.asarray(x >= y), _vec_common_dtype(xp, t_6 + t_7 + t_8 + t_9 + t_10))
