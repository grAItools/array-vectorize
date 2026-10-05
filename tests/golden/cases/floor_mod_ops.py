from array_api_compat import array_namespace

def floor_mod_ops_vec(x):
    """
def floor_mod_ops(x):
    return (x // 2) + (x % 3)
"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.floor_divide(x, 2) + xp.remainder(x, 3)
