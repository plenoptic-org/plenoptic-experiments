#!/usr/bin/env python3

import pathlib
import torch
import plenoptic as po
import imageio.v3 as iio

OUT_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_speed/init_images")

img = po.data.einstein()
for i in range(10):
    po.tools.set_seed(i)
    im_init = torch.rand_like(img)
    im_init = po.tools.convert_float_to_int(po.to_numpy(im_init))
    iio.imwrite(OUT_DIR / f"seed-{i}.png", im_init.squeeze())
