import numpy as np
TEXTURE_DIR = Path("/mnt/ceph/users/xzhao/Datasets")

def _prep(img):
    dim = min(img.shape[-2:])
    n_downsample = int(np.floor(np.log2(dim//224)))
    if n_downsample > 0:
        img = po.process.blur_downsample(img, n_downsample)
    return po.process.center_crop(img, 224)

def brick():
    # brick but not the one in figure 2
    path = TEXTURE_DIR / "cropped1024-selected/10D_9452.tif"
    return _prep(po.load_images(path, as_gray=False))

def straw():
    # straw but not the one in figure 2
    path = TEXTURE_DIR / "dtd/images/matted/matted_0085.jpg"
    return _prep(po.load_images(path, as_gray=False))

def cracked():
    path = TEXTURE_DIR / "dtd/images/cracked/cracked_0077.jpg"
    return _prep(po.load_images(path, as_gray=False))
