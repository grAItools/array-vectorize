from array_api_compat import array_namespace

def relu_with_else_vec(x):
    """
    def relu_with_else(x):
        if x > 0:  # noqa: SIM108
            r = x
        else:
            r = 0.0
        return r
    """
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    r_1 = x
    r_2 = 0.0
    r = xp.where(x > 0, r_1, r_2)
    return r
