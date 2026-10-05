from array_api_compat import array_namespace

def pow_op_vec(x):
    """
    def pow_op(x):
        return x**2 + 2.0**x
    """
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.pow(x, 2) + xp.pow(2.0, x)
