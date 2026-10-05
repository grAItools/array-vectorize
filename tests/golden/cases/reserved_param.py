from array_api_compat import array_namespace

def reserved_param_vec(xp):
    """
    def reserved_param(xp):
        return xp + 1
    """
    xp_1 = array_namespace(xp)
    return xp + 1
