from array_api_compat import array_namespace

def boolop_numeric_vec(x, y):
    """def boolop_numeric(x, y):
    return x and y"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    y = xp.asarray(y)
    return xp.where(x != 0, y, x)
