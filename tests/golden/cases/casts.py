from array_api_compat import array_namespace

def casts_vec(x):
    """def casts(x):
    return int(x) + float(x) + bool(x) + math.trunc(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    t_1 = xp.asarray(x)
    t_2 = xp.astype(t_1, xp.int64)
    t_3 = xp.astype(t_1, xp.float64)
    return t_2 + t_3 + xp.astype(xp.asarray(xp.astype(t_1, xp.bool)), _vec_common_dtype(xp, t_2 + t_3)) + t_2
