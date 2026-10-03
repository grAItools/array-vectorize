from array_api_compat import array_namespace

def bitwise_ops_vec(x, y):
    """def bitwise_ops(x, y):
    return (x & y) | (x ^ 3) << 1 >> 2"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    return x & y | (x ^ 3) << 1 >> 2
