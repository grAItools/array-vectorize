from array_api_compat import array_namespace

def cse_opportunity_vec(x):
    """def cse_opportunity(x):
    return math.sqrt(x) * math.sqrt(x) + math.sqrt(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    t_1 = xp.sqrt(x)
    return t_1 * t_1 + t_1
