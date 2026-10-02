from array_api_compat import array_namespace

def loop_helper_caller_vec(x):
    """def loop_helper_caller(x):
    s = 0.0
    for i in range(3):
        s = s + helper_inner(x)
    return s"""
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    s = 0.0
    s_1 = s
    for i in range(0, 3):
        s_1 = s_1 + helper_inner_vec(x)
    return s_1
