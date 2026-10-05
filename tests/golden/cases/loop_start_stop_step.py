from array_api_compat import array_namespace

def loop_start_stop_step_vec(x):
    """
    def loop_start_stop_step(x):
        s = 1.0
        for i in range(2, 8, 3):
            s = s * i
        return s + x
    """
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    s = 1.0
    s_1 = s
    for i in range(2, 8, 3):
        s_1 = s_1 * i
    return s_1 + x
