from array_api_compat import array_namespace

def math_two_arg_vec(x, y):
    """def math_two_arg(x, y):
    return math.atan2(x, y) + math.hypot(x, y) + math.copysign(x, y)"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    y = xp.asarray(y)
    return xp.atan2(x, y) + xp.hypot(x, y) + xp.copysign(x, y)
