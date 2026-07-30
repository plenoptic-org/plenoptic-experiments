#!/usr/bin/env python3

import re
import torch
import plenoptic as po
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys

def setup(load_path: pathlib.Path, device):
    from synthesize import init_metamer
    model, img, penalty, lmbda, _, seed, n_iter = re.findall(r"model-([A-Za-z]+)_img-([A-Za-z_]+)_penalty-([A-Za-z-]+)_lambda-([0-9\.e-]+)_([0-9a-z]+)_seed-([0-9]+)_iter-([0-9]+)", load_path.name)[0]
    seed = int(seed)
    lmbda = float(lmbda)
    n_iter = int(n_iter)

    print('initializing')
    met, optim, opt_kwargs = init_metamer(img, model, penalty, lmbda,
                                          2, seed, device)
    print("loading")
    met.load(load_path, map_location=device)
    return met

def compute(met, device):
    saved_mets = met.saved_metamer.to(device)
    print("computing")
    try:
        met_reps = torch.func.vmap(met.model.forward)(saved_mets)
    except RuntimeError:
        met_reps = torch.stack([met.model(s) for s in saved_mets])
    try:
        met_loss = torch.func.vmap(met.loss_function, (0, None))(met_reps, met.target_representation)
    except RuntimeError:
        met_loss = torch.stack([met.loss_function(r, met.target_representation)
                                for r in met_reps])
    try:
        met_penalty = torch.func.vmap(met.penalty_function)(saved_mets)
    except RuntimeError:
        met_penalty = torch.stack([met.penalty_function(s) for s in saved_mets])
    return met_loss, met_penalty

def plot(met, met_loss, met_penalty, save_path):
    # for plotting
    print("plotting")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[1, 0].semilogy(met.store_progress * np.arange(len(met_loss)),
                        po.to_numpy(met_loss), label="Loss")
    axes[1, 0].semilogy(met.store_progress * np.arange(len(met_penalty)),
                        po.to_numpy(met_penalty), label="Penalty")
    axes[1, 0].legend()
    po.plot.imshow(met.image.mean(0, keepdim=True), ax=axes[0, 0], title="Target image", vrange=(0, 1))
    po.plot.imshow(met.saved_metamer[0, :1], ax=axes[0, 1], title="Initial image 0", vrange=(0, 1))
    po.plot.imshow(met.saved_metamer[0, 1:], ax=axes[1, 1], title="Initial image 1", vrange=(0, 1))
    po.plot.imshow(met.metamer[:1], ax=axes[0, 2], title="Metamer 0", vrange=(0, 1))
    po.plot.imshow(met.metamer[1:], ax=axes[1, 2], title="Metamer 1", vrange=(0, 1))
    fig.savefig(save_path)


def animate(met, save_path: pathlib.Path):
    print("animating")

    po.synth.metamer.animate(met, batch_idx=0).save(save_path)
    po.synth.metamer.animate(met, batch_idx=1).save(save_path.with_name(f"{save_path.stem}-1.mp4"))


if __name__ == '__main__':
    print("starting")
    in_file = pathlib.Path(sys.argv[1])
    print(sys.argv)
    try:
        device = torch.device(sys.argv[2])
    except RuntimeError:
        device = torch.device(int(sys.argv[2]))
    except IndexError:
        device = torch.device("cpu")
    met = setup(in_file, device)
    met_loss, met_penalty = compute(met, device)
    met.to("cpu")
    plot(met, met_loss, met_penalty, in_file.with_suffix(".svg"))
    animate(met, in_file.with_name(f"{in_file.stem}-0.mp4"))
