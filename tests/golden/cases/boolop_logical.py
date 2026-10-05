from array_api_compat import array_namespace

def boolop_logical_vec(x, y):
    """
    def boolop_logical(x, y):
        return ((x > 0) and (y > 0)) or not (x < 0)
    """
    xp = array_namespace(x, y)
    return xp.logical_or(xp.logical_and(x > 0, y > 0), xp.logical_not(x < 0))
