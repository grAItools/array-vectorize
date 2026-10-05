from array_api_compat import array_namespace

def closure_array_vec(x, *, ARR=None):
    """
    def closure_array(x):
        return x + ARR
    """
    xp = array_namespace(x, ARR)
    return x + ARR
