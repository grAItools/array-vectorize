from array_api_compat import array_namespace

def add_vec(x, y):
    """
def add(x, y):
    return x + y
"""
    xp = array_namespace(*[a for a in (x, y) if hasattr(a, '__array_namespace__')])
    return x + y
