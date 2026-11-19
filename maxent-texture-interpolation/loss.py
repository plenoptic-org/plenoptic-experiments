from maxent_texture import load_pretrained
import numpy as np
TEXTURE_DIR = Path("/mnt/ceph/users/xzhao/Datasets")

class MaxEntTexture(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = load_pretrained()

    def forward(self, *args, **kwargs):
        return self.model.stat_model(*args, **kwargs)

def _prep(img):
    dim = min(img.shape[-2:])
    img = po.process.blur_downsample(img, int(np.floor(np.log2(dim//128))))
    img = po.process.center_crop(img, 128)
    return po.process.rescale(img, -1, 1)

def _straw():
    # straw but not the one in figure 2
    path = TEXTURE_DIR / "dtd/images/matted/matted_0085.jpg"
    return _prep(po.load_images(path, as_gray=False))

def _cracked():
    path = TEXTURE_DIR / "dtd/images/cracked/cracked_0077.jpg"
    return _prep(po.load_images(path, as_gray=False))

class InterpolationLoss(torch.nn.Module):
    def __init__(self, percent):
        super().__init__()
        model = MaxEntTexture().eval()
        img = torch.cat([_straw(), _cracked()])
        weights = torch.as_tensor([percent, 1-percent])
        print(f"Weighted sum using {weights=}")
        target_rep = (weights.unsqueeze(1) * model(img)).sum(0)
        self.register_buffer("target_rep", target_rep)
        self.model = model

    def forward(self, x):
        return po.loss.l2_norm(self.target_rep, self.model(x)).pow(2)

def loss_20_percent():
    loss = InterpolationLoss(.2)
    po.remove_grad(loss)
    return loss

def loss_40_percent():
    loss = InterpolationLoss(.4)
    po.remove_grad(loss)
    return loss

def loss_60_percent():
    loss = InterpolationLoss(.6)
    po.remove_grad(loss)
    return loss

def loss_80_percent():
    loss = InterpolationLoss(.8)
    po.remove_grad(loss)
    return loss
