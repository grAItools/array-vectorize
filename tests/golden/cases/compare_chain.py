from array_api_compat import array_namespace

def compare_chain_vec(x):
    """
    def compare_chain(x):
        return 0 < x < 1
    """
    xp = array_namespace(x)
    return xp.logical_and(0 < x, x < 1)
