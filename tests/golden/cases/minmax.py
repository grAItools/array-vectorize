from array_api_compat import array_namespace

def minmax_vec(x, y):
    """def minmax(x, y):
    return min(x, y) + max(x, y, 3.0)"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    t_1 = xp.asarray(x)
    t_2 = xp.asarray(y)
    return xp.minimum(t_1, t_2) + xp.maximum(xp.maximum(t_1, t_2), 3.0)
