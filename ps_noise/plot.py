#!/usr/bin/env python3

import re
import pandas as pd
import pathlib
import torch
import scipy.io as sio
import plenoptic as po
import matplotlib.pyplot as plt
from plenoptic.data.fetch import fetch_data

SYNTH_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/ps_noise/")
model = po.simul.PortillaSimoncelli((256, 256))
model.to(torch.float64).to(0)
imgs = {"curie": po.data.curie(), "einstein": po.data.einstein()}
imgs = {k: v.to(0).to(torch.float64) for k, v in imgs.items()}

for f in SYNTH_DIR.glob("*pt"):
    save_path = f.with_suffix(".svg")
    if save_path.exists():
        continue

    history, noise, weight, device, img_name, seed = re.findall("LBFGS-([0-9]+)_noise-([0-9]+)_wt-([0-9]+)_([0-9a-z]+)_([a-z]+)_seed-([0-9]+)", f.name)[0]

    met = torch.load(f, map_location=torch.device(0))
    met_img = met["metamer"]
    synth_time = met["synth_time"] - met["initial_time"]
    img = imgs[img_name]

    po.tools.set_seed(int(seed))
    init_img = float(noise) * torch.randn_like(img) + img

    fig, axes = plt.subplots(1, 4, figsize=(32, 5), gridspec_kw={"width_ratios": [1, 1, 1, 3.1]})
    loss = po.tools.optim.l2_norm(model(img), model(met_img))
    for im, ax, t in zip([img, init_img, met_img], axes, ["Original image", f"Initial noise-{noise}\nseed-{seed}", f"Metamer\n{loss}"]):
        po.imshow(im, ax=ax, title=t)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    model.plot_representation(model(img) - model(met_img), ax=axes[-1], ylim=False)
    axes[-1].set_title(f"PS(img) - PS(metamer)\nHighpass weight: {weight}", y=1.05)
    fig.suptitle(f"Device: {device}, duration: {synth_time}", fontsize='xx-large', y=1.1)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
