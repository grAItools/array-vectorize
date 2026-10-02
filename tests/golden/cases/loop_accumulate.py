from array_api_compat import array_namespace

def loop_accumulate_vec(x):
    """def loop_accumulate(x):
    s = 0.0
    for i in range(4):  # noqa: B007
        s = s + x
    return s"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    x = xp.asarray(x)
    s = 0.0
    s_1 = s
    for i in range(0, 4):
        s_1 = xp.astype(xp.asarray(s_1), xp.float64) + x
    return s_1
