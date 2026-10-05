from array_api_compat import array_namespace

def from_import_vec(x):
    """
    def from_import(x):
        return fexp(x)
    """
    xp = array_namespace(x)
    return xp.exp(xp.astype(xp.asarray(x), xp.float64))
