from array_api_compat import array_namespace

def clamp_vec(x, lo=0.0, hi=1.0):
    """def clamp(x, lo=0.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x"""
    xp = array_namespace(*[a for a in (x, lo, hi) if hasattr(a, '__array_namespace__')])
    return xp.where(x < lo, lo, xp.where(x > hi, hi, x))
