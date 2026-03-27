#!/usr/bin/env python3

import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import plenoptic as po
import torch
import pathlib

plt.close("all")
BASE_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/penalty_visual_diversity")

files = [
    BASE_DIR / "model-LGC_img-einstein_penalty-none_lambda-0.001_cpu_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein_penalty-nlpd-sse_lambda-0.001_cpu_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein_penalty-spyr-expmaskall_lambda-1e-02_0_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein_penalty-spyr-expmaskvert_lambda-1e-04_0_seed-0_iter-6000.pt",
    BASE_DIR / "model-LGC_img-einstein_penalty-spyr-expmaskhigh_lambda-1e-04_0_seed-0_iter-6000.pt",
]

imgs = []

for f in files:
    imgs.append(torch.load(f)["_metamer"].to(0))

imgs = torch.stack(imgs).clamp(0, 1)

synth_dist = torch.stack([po.metric.nlpd(i[:1], i[1:]).squeeze() for i in imgs]).cpu()

ax = sns.barplot(synth_dist)
ax.figure.savefig("nlpd_dist.svg")

imgs = imgs.flatten(0, 1)

all_dist = torch.zeros(len(imgs), len(imgs))

for i, j in itertools.combinations(range(len(imgs)), 2):
    all_dist[j, i] = po.metric.nlpd(imgs[i].unsqueeze(0), imgs[j].unsqueeze(0)).squeeze()
    all_dist[i, j] = torch.nan

# all_dist = all_dist - all_dist[1, 0]

ax = sns.heatmap(all_dist, cmap="RdBu_r", center=0)
ax.figure.savefig("nlpd_all.svg")
