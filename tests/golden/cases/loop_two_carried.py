from array_api_compat import array_namespace

def loop_two_carried_vec(x):
    """
    def loop_two_carried(x):
        a = 0.0
        b = 1.0
        for i in range(3):  # noqa: B007
            a = a + x
            b = b * 2.0
        return a + b
    """
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    a = 0.0
    b = 1.0
    a_1 = a
    b_1 = b
    for i in range(0, 3):
        a_1 = a_1 + x
        b_1 = b_1 * 2.0
    return a_1 + b_1
