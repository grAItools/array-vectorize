from array_api_compat import array_namespace

def math_floor_ceil_vec(x):
    """
    def math_floor_ceil(x):
        return math.floor(x) + math.ceil(x)
    """
    xp = array_namespace(x)
    t_1 = xp.asarray(x)
    t_2 = xp.astype(t_1, xp.float64)
    return xp.floor(t_2) + xp.ceil(t_2)
