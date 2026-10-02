from array_api_compat import array_namespace

def relu_vec(x):
    """def relu(x):
    if x > 0:
        return x
    return 0.0"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    return xp.where(x > 0, x, 0.0)
