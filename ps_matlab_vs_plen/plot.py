#!/usr/bin/env python3

import pandas as pd
import pathlib
import torch
import scipy.io as sio
import plenoptic as po
import matplotlib.pyplot as plt
from plenoptic.data.fetch import fetch_data
IMG_DIR = fetch_data("portilla_simoncelli_images.tar.gz")

INIT_IMG_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_matlab_vs_plen/init_images")
OUT_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_matlab_vs_plen/plenoptic_results_weighted")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv("results/all_loss.csv")
imgs ={k: po.load_images(IMG_DIR / f"{k}.jpg").to(torch.float64)
       for k in df.img.unique()}
init_imgs ={k: po.load_images(INIT_IMG_DIR / f"{k}.png").to(torch.float64)
            for k in df.init_img.unique()}

df.res_highpass_weight = df.res_highpass_weight.astype(str)
for n, gb in df.groupby(["optimizer", "res_highpass_weight", "img", "init_img", "device"]):
    save_path = OUT_DIR / f"metamers_{'_'.join(n)}.svg"
    if save_path.exists():
        continue
    mets = [init_imgs[n[3]]]
    title = ["Initial"]
    for m, g in sorted(gb.groupby("synth_iters"), key=lambda x: int(x[0].replace('iter-', ''))):
        assert g.output_path.nunique() == 1
        met = g.output_path.unique()[0]
        title.append(m)
        if met.endswith(".mat"):
            met = sio.loadmat(met)
            met = torch.from_numpy(met["res"] / 255).unsqueeze(0).unsqueeze(0)
        else:
            met = torch.load(met, map_location="cpu")
            met = met["metamer"]
        mets.append(met)
    mets = torch.cat(mets + [imgs[n[2]]], dim=0)
    fig = po.imshow(mets, 'auto1', title=title + ["Original image"])
    fig.suptitle(n, fontsize='xx-large', y=1.1)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
