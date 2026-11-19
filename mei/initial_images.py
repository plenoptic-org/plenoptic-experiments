def rand():
    return torch.rand((1, 1, 128, 128))

# def rand_small():
#     return (0.1 * torch.rand((1, 1, 128, 128))) + 0.5

# def randn():
#     return 0.1 * torch.randn((1, 1, 128, 128))

def randn_ctr():
    return (0.1 * torch.randn((1, 1, 128, 128))) + 0.5
