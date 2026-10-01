from array_api_compat import array_namespace

def abs_round_vec(x):
    """def abs_round(x):
    return abs(x) + round(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.abs(x) + xp.round(x)
