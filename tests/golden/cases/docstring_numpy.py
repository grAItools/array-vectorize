from array_api_compat import array_namespace

def docstring_numpy_vec(x, y):
    '''
    (array-vectorized) Add two values, scaled.

    Parameters
    ----------
    x : float
        first value
    y : float
        second value

    Returns
    -------
    float
        the scaled sum

    Notes
    -----
    Vectorized by array-vectorize from this scalar original::

        def docstring_numpy(x, y):
            """Add two values, scaled.

            Parameters
            ----------
            x : float
                first value
            y : float
                second value

            Returns
            -------
            float
                the scaled sum
            """
            return (x + y) * SCALE
    '''
    xp = array_namespace(x, y)
    return (x + y) * 3.0
