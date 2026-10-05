from array_api_compat import array_namespace

def arith_ops_vec(x, y):
    """
    def arith_ops(x, y):
        return (x + y - x * y) / (x * x + y * y)
    """
    xp = array_namespace(x, y)
    return (x + y - x * y) / (x * x + y * y)
