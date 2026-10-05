from array_api_compat import array_namespace

def both_branches_assign_vec(x):
    """
def both_branches_assign(x):
    if x > 0:
        y = x
        z = 1.0
    else:
        y = -x
        z = 2.0
    return y + z
"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    y_1 = x
    z_1 = 1.0
    y_2 = -x
    z_2 = 2.0
    y = xp.where(x > 0, y_1, y_2)
    z = xp.where(x > 0, z_1, z_2)
    return y + z
