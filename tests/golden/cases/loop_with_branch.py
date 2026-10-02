from array_api_compat import array_namespace

def loop_with_branch_vec(x):
    """def loop_with_branch(x):
    s = 0.0
    for i in range(4):  # noqa: B007
        if x > 0:  # noqa: SIM108
            s = s + x
        else:
            s = s - 1.0
    return s"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    s = 0.0
    s_1 = s
    for i in range(0, 4):
        s_5 = xp.astype(xp.asarray(s_1), xp.float64) + x
        s_6 = xp.astype(xp.asarray(s_1), xp.float64) - 1.0
        s_7 = xp.where(x > 0, s_5, s_6)
        s_1 = s_7
    return s_1
