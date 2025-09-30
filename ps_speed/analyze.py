#!/usr/bin/env python3

import plenoptic as po
from plenoptic.data.fetch import fetch_data
import re
import torch
import pandas as pd
import scipy.io as sio
import sys
import pathlib

IMG_DIR = fetch_data("portilla_simoncelli_images.tar.gz")

met = pathlib.Path(sys.argv[1])
ps = po.simul.PortillaSimoncelli((256, 256))
logfile = met.with_suffix(".log")
img = re.findall("(fig[0-9]+[a-z])_", met.name)[0]
init_img = re.findall("(seed-[0-9]+)_", met.name)[0]
n_iters = re.findall("_(iter-[0-9]+)", met.name)[0]
save_path = met.with_suffix(".csv")

data = {
    "img": img,
    "init_img": init_img,
    "output_path": str(met),
    "synth_iters": n_iters,
    "res_highpass_weight": 1,
}

if met.suffix == ".mat":
    met = sio.loadmat(met)
    init_time = float(met["initToc"].squeeze())
    synth_time = float(met["synthToc"].squeeze())
    met = torch.from_numpy(met["res"] / 255).unsqueeze(0).unsqueeze(0)
    data["device"] = "cpu"
    data["optimizer"] = "matlab"
else:
    if "LBFGS" in met.name:
        history, weight, device = re.findall("LBFGS-([0-9]+)_wt-([0-9]+)_([0-9a-z]+)_fig", met.name)[0]
        data["optimizer"] = f"LBFGS-{history}"
        data["res_highpass_weight"] = weight
    else:
        device = re.findall("Adam_([0-9a-z]+)_fig", met.name)[0]
        data["optimizer"] = "Adam"
    try:
        # if this succeeds, then it was a GPU
        int(device)
        device = "cuda"
    except ValueError:
        # then this is a string, so leave it
        pass
    data["device"] = device
    met = torch.load(met, map_location="cpu")
    init_time = met["initial_time"]
    synth_time = met["synth_time"]
    met = met["metamer"]

synth_time = synth_time - init_time
with open(logfile) as f:
    total_time = float(f.readlines()[-1])
img = po.load_images(IMG_DIR / f"{img}.jpg").to(torch.float64)

rep = ps(img)
met_rep = ps(met)

data.update({
    "synth_only_time": synth_time,
    "init_time": init_time,
    "total_time": total_time,
})


def mse(x, y):
    return (x-y).pow(2).nanmean().item()
def sse(x, y):
    return (x-y).pow(2).nansum().item()
def l2_norm(x, y):
    return (x-y).pow(2).nansum().sqrt().item()


df = []
funcs = []
rep_dict = ps.convert_to_dict(rep)
met_rep_dict = ps.convert_to_dict(met_rep)

for func in [mse, sse, l2_norm]:
    d = data.copy()
    d["loss"] = func(rep, met_rep)
    d["loss_type"] = "overall"
    d["loss_func"] = func.__name__
    df.append(d)
    for k in rep_dict.keys():
        d = data.copy()
        d["loss"] = func(rep_dict[k], met_rep_dict[k])
        d["loss_type"] = k
        d["loss_func"] = func.__name__
        df.append(d)

df = pd.DataFrame(df)
df.to_csv(save_path, index=False)
