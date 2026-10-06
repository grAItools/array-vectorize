from array_api_compat import array_namespace

def closure_array_vec(x, *, ARR=None):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def closure_array(x):
                return x + ARR
    """
    xp = array_namespace(x, ARR)
    return x + ARR
