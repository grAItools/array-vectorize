from array_api_compat import array_namespace

def cse_opportunity_vec(x):
    """
def cse_opportunity(x):
    return math.sqrt(x) * math.sqrt(x) + math.sqrt(x)
"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    t_1 = xp.asarray(x)
    t_2 = xp.astype(t_1, xp.float64)
    t_3 = xp.sqrt(t_2)
    return t_3 * t_3 + t_3
