from array_api_compat import array_namespace

def boolop_numeric_vec(x, y):
    """
    def boolop_numeric(x, y):
        return x and y
    """
    xp = array_namespace(x, y)
    return xp.where(x != 0, y, x)
