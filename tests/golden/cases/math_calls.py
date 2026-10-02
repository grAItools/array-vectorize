from array_api_compat import array_namespace

def math_calls_vec(x):
    """def math_calls(x):
    return math.sqrt(x) + math.exp(x) + math.log1p(x) + math.sin(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    return xp.sqrt(x) + xp.exp(x) + xp.log1p(x) + xp.sin(x)
