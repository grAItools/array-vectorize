from array_api_compat import array_namespace

def kwonly_defaults_vec(x, *, scale=2.0, bias=0.5):
    """
    def kwonly_defaults(x, *, scale=2.0, bias=0.5):
        return x * scale + bias
    """
    xp = array_namespace(*[a for a in (x, scale, bias) if hasattr(a, '__array_namespace__')])
    return x * scale + bias
