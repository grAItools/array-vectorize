from array_api_compat import array_namespace

def compare_ops_vec(x, y):
    """def compare_ops(x, y):
    return (x == y) + (x != y) + (x < y) + (x <= y) + (x > y) + (x >= y)"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    return xp.astype(xp.asarray(x == y), xp.int64) + xp.astype(xp.asarray(x != y), xp.int64) + xp.astype(xp.asarray(x < y), xp.int64) + xp.astype(xp.asarray(x <= y), xp.int64) + xp.astype(xp.asarray(x > y), xp.int64) + xp.astype(xp.asarray(x >= y), xp.int64)
