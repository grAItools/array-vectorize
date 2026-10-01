from array_api_compat import array_namespace

def casts_vec(x):
    """def casts(x):
    return int(x) + float(x) + bool(x) + math.trunc(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    t_1 = xp.astype(x, xp.int64)
    return t_1 + xp.astype(x, xp.float64) + xp.astype(x, xp.bool) + t_1
