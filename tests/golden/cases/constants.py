from array_api_compat import array_namespace

def constants_vec(x):
    """
def constants(x):
    return x * math.pi + math.e + math.inf - math.nan
"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return x * 3.141592653589793 + 2.718281828459045 + xp.inf - xp.nan
