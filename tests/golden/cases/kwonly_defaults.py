from array_api_compat import array_namespace

def kwonly_defaults_vec(x, *, scale=2.0, bias=0.5):
    """
    def kwonly_defaults(x, *, scale=2.0, bias=0.5):
        return x * scale + bias
    """
    xp = array_namespace(x, scale, bias)
    return x * scale + bias
