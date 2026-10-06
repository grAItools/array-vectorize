from array_api_compat import array_namespace

def merge_with_prior_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def merge_with_prior(x):
                r = 0.0
                if x > 0:
                    r = x
                return r
    """
    xp = array_namespace(x)
    r = 0.0
    r_1 = x
    r_2 = xp.where(x > 0, r_1, r)
    return r_2
