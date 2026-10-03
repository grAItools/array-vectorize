from array_api_compat import array_namespace

def compare_chain_vec(x):
    """def compare_chain(x):
    return 0 < x < 1"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.logical_and(0 < x, x < 1)
