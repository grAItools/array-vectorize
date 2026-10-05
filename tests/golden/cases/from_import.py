from array_api_compat import array_namespace

def from_import_vec(x):
    """
    def from_import(x):
        return fexp(x)
    """
    xp = array_namespace(*[a for a in (x,) if hasattr(a, '__array_namespace__')])
    return xp.exp(xp.astype(xp.asarray(x), xp.float64))
