from array_api_compat import array_namespace

def add_vec(x, y):
    """
    def add(x, y):
        return x + y
    """
    xp = array_namespace(x, y)
    return x + y
