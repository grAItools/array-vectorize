from array_api_compat import array_namespace

def reserved_param_vec(xp_):
    """def reserved_param(xp):
    return xp + 1"""
    xp = array_namespace(*[a for a in (xp_,) if hasattr(a, '__array_namespace__')])
    return xp_ + 1
