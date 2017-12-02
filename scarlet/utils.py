def invert_with_zeros(x):
    mask = (x == 0)
    x[~mask] = 1./x[~mask]
    x[mask] = -1
    return x
