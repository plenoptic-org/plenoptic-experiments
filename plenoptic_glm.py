#!/usr/bin/env python3

import plenoptic as po
import torch
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.close("all")

def jax_to_torch(x, n_unsqueeze=0):
    x = torch.from_numpy(copy.copy(np.asarray(x)))
    for _ in range(n_unsqueeze):
        x = x.unsqueeze(0)
    return x

class GLM(torch.nn.Module):
    def __init__(self, weight_shape=None, weight=None, bias=None, link_func="exp"):
        super().__init__()
        if weight_shape is not None and weight is not None:
            raise ValueError("Exactly one of weight_shape and weight must be set!")
        if weight_shape is None and weight is None:
            raise ValueError("Exactly one of weight_shape and weight must be set!")
        if weight_shape is None:
            weight_shape = weight.shape
            dtype = weight.dtype
        else:
            dtype = torch.float32
        if len(weight_shape) == 1:
            self.conv = torch.nn.Conv1d(1, 1, weight_shape, dtype=dtype)
        elif len(weight_shape) == 2:
            self.conv = torch.nn.Conv2d(1, 1, weight_shape, dtype=dtype)
        elif len(weight_shape) == 3:
            self.conv = torch.nn.Conv3d(1, 1, weight_shape, dtype=dtype)
        state_dict = {}
        if weight is not None:
            state_dict["conv.weight"] = weight.unsqueeze(0).unsqueeze(0)
        if bias is not None:
            state_dict["conv.bias"] = bias
        if link_func == "exp":
            self.link_func = torch.exp
        else:
            raise ValueError(f"Don't know how to handle {link_func=}")
        self.load_state_dict(state_dict)

    def forward(self, x, **kwargs):
        return self.link_func(self.conv(x, **kwargs))

    @classmethod
    def from_nemos_glm(cls, coeffs_npz, nemos_key):
        # nemos convention is reverse of torch's
        weight = jax_to_torch(coeffs_npz[f"{nemos_key}_coef_"][::-1])
        bias = jax_to_torch(coeffs_npz[f"{nemos_key}_intercept_"])
        link_func = coeffs_npz[f"{nemos_key}_link_func"]
        return cls(weight=weight, bias=bias, link_func=link_func)

def plot(met, met_penalty, save_path="tmp.svg"):
    gs = mpl.gridspec.GridSpec(4, 2, width_ratios=[1, 3])
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    ax = fig.add_subplot(gs[1:3, 0])
    ax.plot(po.to_numpy(met.model.conv.weight.squeeze()))
    ax.set_title("Filter")
    ax = fig.add_subplot(gs[:2, 1])
    ax.set_title("Stimuli")
    ax.plot(met.image.squeeze(), label="Real stimulus")
    ax.plot(po.to_numpy(met.metamer.squeeze()), "--", label="Metamer")
    ax.plot(po.to_numpy(met_penalty.metamer.squeeze()), "--", label="Metamer with penalty")
    ax.legend()
    ax.set(xlim=(0, 200))
    model_stim = met.model(met.image).squeeze()
    model_stim = torch.cat([torch.nan * torch.zeros(stim.shape[-1]-len(model_stim)), model_stim])
    ax = fig.add_subplot(gs[2:, 1])
    ax.set_title("Model response")
    ax.plot(po.to_numpy(model_stim), label="Real stimulus")
    model_met = met.model(met.metamer).squeeze()
    model_met = torch.cat([torch.nan * torch.zeros(stim.shape[-1]-len(model_met)), model_met])
    ax.plot(po.to_numpy(model_met), "--", label="Metamer")
    model_met = met_penalty.model(met_penalty.metamer).squeeze()
    model_met = torch.cat([torch.nan * torch.zeros(stim.shape[-1]-len(model_met)), model_met])
    ax.plot(po.to_numpy(model_met), "--", label="Metamer with penalty")
    ax.legend()
    ax.set(xlim=(0, 200))
    fig.savefig(save_path, bbox_inches="tight")

def run_met(stim, model):
    po.remove_grad(model)
    model.eval()

    met = po.Metamer(stim, model, penalty_lambda=0)
    met.setup(optimizer=torch.optim.LBFGS)
    met.synthesize(1000, stop_criterion=1e-20)

    met_penalty = po.Metamer(stim, model, penalty_function=lambda x: po.regularize.penalize_range(x, (-.5, .5)))
    met_penalty.setup(optimizer=torch.optim.LBFGS)
    met_penalty.synthesize(1000, stop_criterion=1e-20)

    return met, met_penalty


# load in models

coeffs = np.load("nemos_coeffs.npz")
simulations = np.load("nemos_simulations.npz")

glm = GLM.from_nemos_glm(coeffs, "stim_alone")
glm_spk = GLM.from_nemos_glm(coeffs, "stim_spike")

# test models

for i in range(simulations["n_simulations"]):
    sim_input = jax_to_torch(simulations[f"input_{i}"], 2)
    sim_output = jax_to_torch(simulations[f"output_stim_{i}"], 2)[..., 19:]
    assert torch.allclose(sim_output, glm(sim_input))
    # zeros_spk has another time point of nans because conv_kwargs isn't set
    sim_output = jax_to_torch(simulations[f"output_stim_spk_{i}"], 2)[..., 20:]
    assert torch.allclose(sim_output, glm_spk(sim_input)[..., 1:])


# do plenoptic
stim = jax_to_torch(coeffs["stimulus"], 2)[..., :200]
met, met_penalty = run_met(stim, glm)

# plots
plot(met, met_penalty, "glm.svg")

# do plenoptic with other model
met, met_penalty = run_met(stim, glm_spk)

# plots
plot(met, met_penalty, "glm_spk.svg")
