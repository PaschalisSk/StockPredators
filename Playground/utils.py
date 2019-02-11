from numpy.lib.stride_tricks import as_strided as strided


def get_sliding_window(df, W, return2D=0):
    # https://stackoverflow.com/questions/37447347/dataframe-representation-of-a-rolling-window/41406783#41406783
    a = df.values
    s0, s1 = a.strides
    m, n = a.shape
    out = strided(a, shape=(m - W + 1, W, n), strides=(s0, s0, s1))
    if return2D == 1:
        return out.reshape(a.shape[0] - W + 1, -1)
    else:
        return out
