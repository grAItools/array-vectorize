from array_api_compat import array_namespace

def ssa_rebind_vec(x):
    """
    def ssa_rebind(x):
        y = x + 1
        y = y * 2
        y += 3
        return y
    """
    xp = array_namespace(x)
    y = x + 1
    y_1 = y * 2
    y_2 = y_1 + 3
    return y_2
