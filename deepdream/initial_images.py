def randn():
    return 0.01 * torch.randn((1, 3, 224, 224))

def randn_add():
    return 0.5 + (0.01 * torch.randn((1, 3, 224, 224)))

def rand_small():
    return 0.5 + (0.1 * torch.rand((1, 3, 224, 224)))

def rand():
    return torch.rand((1, 3, 224, 224))
