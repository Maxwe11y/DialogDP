

def max_min_scale(x, new_min, new_max, eps=1e-12):
    x_max = x.max()
    x_min = x.min()
    return (x - x_min) / (x_max - x_min + eps) * (new_max - new_min) + new_min