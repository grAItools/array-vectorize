from array_api_compat import array_namespace

def nested_loops_vec(x):
    """
    def nested_loops(x):
        s = 0.0
        for i in range(3):  # noqa: B007
            for j in range(2):  # noqa: B007  # noqa: B007
                s = s + x
        return s
    """
    xp = array_namespace(x)
    s = 0.0
    s_1 = s
    for i in range(0, 3):
        s_2 = s_1
        for j in range(0, 2):
            s_2 = s_2 + x
            s_1 = s_2
    return s_1
