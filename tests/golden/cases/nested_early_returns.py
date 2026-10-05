from array_api_compat import array_namespace

def nested_early_returns_vec(x):
    """
    def nested_early_returns(x):
        if x > 0:
            if x > 10:
                return 1.0
            return 2.0
        return 3.0
    """
    xp = array_namespace(x)
    return xp.where(x > 0, xp.where(x > 10, 1.0, 2.0), 3.0)
