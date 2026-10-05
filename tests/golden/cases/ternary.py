from array_api_compat import array_namespace

def ternary_vec(x):
    """
    def ternary(x):
        return x * 2 if x > 0 else x / 2
    """
    xp = array_namespace(x)
    return xp.where(x > 0, x * 2, x / 2)
