# def randn_rescale(image):
#     init_img = 0.01 * (torch.rand_like(image) - 0.5)
#     return po.process.rescale(init_img, 0, 1)

def randn(image):
    init_img = 0.01 * torch.randn_like(image)
    return 0.5 + init_img

def rand(image):
    init_img = torch.rand_like(image)
    return init_img

def rand_small(image):
    init_img = 0.1 * torch.rand_like(image)
    return 0.5 + init_img
