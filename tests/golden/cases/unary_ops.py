from array_api_compat import array_namespace

def unary_ops_vec(x):
    """def unary_ops(x):
    return -x + +x - ~x"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return -x + +x - ~x
