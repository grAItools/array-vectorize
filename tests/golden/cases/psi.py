from array_api_compat import array_namespace

def psi_vec(x):
    """def psi(x):
    if x < 0:
        return 0.0
    return x * math.exp(-x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.where(x < 0, 0.0, x * xp.exp(xp.astype(xp.asarray(-x), xp.float64)))
