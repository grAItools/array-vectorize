from array_api_compat import array_namespace

def piecewise_vec(x):
    """def piecewise(x):
    if x < -1:
        return -1.0
    elif x > 1:
        return 1.0
    return x"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.where(x < -1, -1.0, xp.where(x > 1, 1.0, x))
