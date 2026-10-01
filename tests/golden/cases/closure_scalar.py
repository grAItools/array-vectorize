from array_api_compat import array_namespace

def closure_scalar_vec(x):
    """def closure_scalar(x):
    return x * SCALE"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return x * 3.0
