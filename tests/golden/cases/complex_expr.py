from array_api_compat import array_namespace

def complex_expr_vec(x, y):
    """def complex_expr(x, y):
    t = math.exp(-(x * x + y * y))
    return t / (1.0 + t)"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    y = xp.asarray(y)
    t = xp.exp(-(x * x + y * y))
    return t / (1.0 + t)
