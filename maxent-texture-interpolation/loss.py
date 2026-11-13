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

def _texture_interpolation(percent):
    model = MaxEntTexture()
    img = torch.cat([_straw(), _cracked()])
    weights = torch.as_tensor([percent, 1-percent]).unsqueeze(1)
    target_rep = (weights * model(img)).sum(0)
    def loss(x):
        return po.loss.l2_norm(target_rep, model(x)).pow(2)
    return loss

def loss_20_percent():
    return _texture_interpolation(.2)

def loss_40_percent():
    return _texture_interpolation(.4)

def loss_60_percent():
    return _texture_interpolation(.6)

def loss_80_percent():
    return _texture_interpolation(.8)
