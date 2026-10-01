from array_api_compat import array_namespace

def math_floor_ceil_vec(x):
    """def math_floor_ceil(x):
    return math.floor(x) + math.ceil(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.floor(x) + xp.ceil(x)
