#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import pathlib
import plenoptic as po


plt.close("all")
BASE_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/penalty_visual_diversity")

files = [
    BASE_DIR / "model-LGC_img-einstein-blur1_penalty-none_lambda-1e+00_0_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein-blur1_penalty-nlpd-sse_lambda-1e-04_0_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein-blur1_penalty-spyr-expmaskall_lambda-1e-02_0_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein-blur1_penalty-spyr-expmasklow_lambda-1e-06_0_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein-blur1_penalty-spyr-expmaskvert_lambda-1e-06_0_seed-0_two-stage_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein-blur1_penalty-pyiqa-dists_exp_lambda-1e-05_0_seed-0_two-stage_iter-6000.pt",
]

imgs = []

for f in files:
    imgs.append(torch.load(f)["_metamer"].to(0).clamp(0, 1))

img = po.process.blur_downsample(po.data.einstein().to(0).to(torch.float64), 1)
imgs = torch.cat([img, *[i[:1] for i in imgs],
                  torch.zeros_like(img), *[i[1:] for i in imgs]])

model = po.models.LuminanceGainControl(
    kernel_size=(31, 31), pad_mode="circular",
    pretrained=True, cache_filt=True
)
model.to(0).to(torch.float64)

model_reps = model(imgs)

titles = ["Target image", "No penalty", "NLPD", "Spectral\nDensity",
          "Low\nFrequency", "Vertical\nPower", "DISTS"]
fig = po.plot.imshow(imgs, col_wrap=len(titles), vrange=(0, 1), title=None)
for ax, t in zip(fig.axes, titles):
    ax.set_title(t, fontsize="large")
fig.axes[len(titles)].set_visible(False)
fig.savefig("metamers.svg", bbox_inches="tight")

fig = po.plot.imshow(model_reps, col_wrap=len(titles), vrange='auto1', title=titles*2)
fig.axes[len(titles)].set_visible(False)
fig.savefig("representations.svg")
