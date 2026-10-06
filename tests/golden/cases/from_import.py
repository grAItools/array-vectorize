from array_api_compat import array_namespace

def from_import_vec(x):
    """
    (array-vectorized) no docstring on the scalar original.

    Notes:
        Vectorized by array-vectorize from this scalar original::

            def from_import(x):
                return fexp(x)
    """
    xp = array_namespace(x)
    return xp.exp(xp.astype(xp.asarray(x), xp.float64))
