def l2_norm_squared():
    return lambda x, y: po.loss.l2_norm(x, y).pow(2)
