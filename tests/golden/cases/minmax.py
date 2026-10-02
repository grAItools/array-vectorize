from array_api_compat import array_namespace

def minmax_vec(x, y):
    """def minmax(x, y):
    return min(x, y) + max(x, y, 3.0)"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    y = xp.asarray(y)
    return xp.minimum(x, y) + xp.maximum(xp.maximum(x, y), xp.asarray(3.0))
