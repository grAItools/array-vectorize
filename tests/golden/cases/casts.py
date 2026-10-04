from array_api_compat import array_namespace

def casts_vec(x):
    """def casts(x):
    return int(x) + float(x) + bool(x) + math.trunc(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    t_1 = xp.asarray(x)
    t_2 = xp.astype(t_1, xp.int64)
    t_3 = xp.astype(t_1, xp.float64)
    t_4 = xp.astype(t_1, xp.bool)
    return xp.astype(xp.asarray(t_2 + t_3), _vec_arith_dtype(xp, 1, t_2 + t_3, t_4)) + xp.astype(xp.asarray(t_4), _vec_arith_dtype(xp, 1, t_2 + t_3, t_4)) + t_2
