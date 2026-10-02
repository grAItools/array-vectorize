from array_api_compat import array_namespace

def casts_vec(x):
    """def casts(x):
    return int(x) + float(x) + bool(x) + math.trunc(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    t_1 = xp.astype(x, xp.int64)
    return t_1 + xp.astype(x, xp.float64) + xp.astype(xp.asarray(xp.astype(x, xp.bool)), xp.float64) + t_1
