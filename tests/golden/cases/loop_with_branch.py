from array_api_compat import array_namespace

def loop_with_branch_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def loop_with_branch(x):
                s = 0.0
                for i in range(4):  # noqa: B007
                    if x > 0:  # noqa: SIM108
                        s = s + x
                    else:
                        s = s - 1.0
                return s
    """
    xp = array_namespace(x)
    s = 0.0
    s_1 = s
    for i in range(0, 4):
        s_2 = s_1 + x
        s_3 = s_1 - 1.0
        s_4 = xp.where(x > 0, s_2, s_3)
        s_1 = s_4
    return s_1
