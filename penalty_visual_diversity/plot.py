#!/usr/bin/env python3

import re
import torch
import plenoptic as po
import numpy as np
import matplotlib.pyplot as plt
from synthesize import init_metamer
import pathlib
import sys

in_file = pathlib.Path(sys.argv[1])
model, img, penalty, lmbda, device, seed, n_iter = re.findall(r"model-([A-Za-z]+)_img-([A-Za-z_]+)_penalty-([A-Za-z-]+)_lambda-([0-9\.]+)_([0-9a-z]+)_seed-([0-9]+)_iter-([0-9]+)", in_file.name)[0]
seed = int(seed)
lmbda = float(lmbda)
n_iter = int(n_iter)

met, optim, opt_kwargs = init_metamer(img, model, penalty, lmbda,
                                      2, seed, "cpu")
met.load(in_file, map_location="cpu")
met_reps = torch.func.vmap(met.model.forward)(met.saved_metamer)
met_loss = torch.func.vmap(met.loss_function, (0, None))(met_reps, met.target_representation)
met_penalty = torch.func.vmap(met.penalty_function)(met.saved_metamer)

po.synth.metamer.animate(met, batch_idx=0).save(f"{in_file.stem}-0.mp4")
po.synth.metamer.animate(met, batch_idx=1).save(f"{in_file.stem}-1.mp4")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes[1, 0].semilogy(met.store_progress * np.arange(len(met_loss)),
                    po.to_numpy(met_loss), label="Loss")
axes[1, 0].semilogy(met.store_progress * np.arange(len(met_penalty)),
                    po.to_numpy(met_penalty), label="Penalty")
axes[1, 0].legend()
po.imshow(met.image.mean(0, keepdim=True), ax=axes[0, 0], title="Target image")
po.imshow(met.saved_metamer[0, :1], ax=axes[0, 1], title="Initial image 0")
po.imshow(met.saved_metamer[0, 1:], ax=axes[1, 1], title="Initial image 1")
po.imshow(met.metamer[:1], ax=axes[0, 2], title="Metamer 0")
po.imshow(met.metamer[1:], ax=axes[1, 2], title="Metamer 1")
fig.savefig(in_file.with_suffix(".svg"))
