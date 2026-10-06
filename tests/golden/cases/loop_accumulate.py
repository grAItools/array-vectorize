from array_api_compat import array_namespace

def loop_accumulate_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def loop_accumulate(x):
                s = 0.0
                for i in range(4):  # noqa: B007
                    s = s + x
                return s
    """
    xp = array_namespace(x)
    s = 0.0
    s_1 = s
    for i in range(0, 4):
        s_1 = s_1 + x
    return s_1
