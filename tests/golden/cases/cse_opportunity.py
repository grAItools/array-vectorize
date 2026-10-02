from array_api_compat import array_namespace

def cse_opportunity_vec(x):
    """def cse_opportunity(x):
    return math.sqrt(x) * math.sqrt(x) + math.sqrt(x)"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    t_1 = xp.asarray(x)
    t_2 = xp.sqrt(t_1)
    return t_2 * t_2 + t_2
