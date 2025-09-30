#!/usr/bin/env python3

# inputs
img = "fig4a.jpg"
seed = 0
history_size = 100
device = "cuda"
weighted = 100

import plenoptic as po
import torch
from plenoptic.data.fetch import fetch_data
IMG_PATH = fetch_data("portilla_simoncelli_images.tar.gz")
img = po.load_images(IMG_PATH / "fig4a.jpg").to(torch.float64).to(device)
img = po.data.curie().to(torch.float64).to(device)
model = po.simul.PortillaSimoncelli(img.shape[-2:])
model.to(device)
po.tools.set_seed(seed)
init_img = torch.rand_like(img)
weights = model.convert_to_dict(torch.ones_like(model(img)))
for k in ["var_highpass_residual"]:
    weights[k] = weighted * torch.ones_like(weights[k])
weights = model.convert_to_tensor(weights)
def loss(x, y):
    return po.tools.optim.l2_norm(weights*x, weights*y)
met = po.synth.Metamer(img, model, loss_function=loss, range_penalty_lambda=0)
met.setup(init_img, optimizer=torch.optim.LBFGS, optimizer_kwargs={"max_iter": 10, "max_eval": 10, "history_size": history_size, "line_search_fn": "strong_wolfe", "lr": 1})
met.synthesize(200)
