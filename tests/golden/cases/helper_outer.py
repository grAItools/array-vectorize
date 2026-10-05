from array_api_compat import array_namespace

def helper_outer_vec(x):
    """
    def helper_outer(x):
        return helper_inner(x) + helper_inner(x * 2.0)
    """
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return helper_inner_vec(x) + helper_inner_vec(x * 2.0)
