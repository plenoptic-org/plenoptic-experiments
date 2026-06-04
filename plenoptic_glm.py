#!/usr/bin/env python3

import plenoptic as po
import torch
import copy
import numpy as np

coeffs = np.load("nemos_coeffs.npz")
simulations = np.load("nemos_simulations.npz")

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

glm = GLM.from_nemos_glm(coeffs, "stim_alone")
glm_spk = GLM.from_nemos_glm(coeffs, "stim_spike")

for i in range(simulations["n_simulations"]):
    sim_input = jax_to_torch(simulations[f"input_{i}"], 2)
    sim_output = jax_to_torch(simulations[f"output_stim_{i}"], 2)[..., 19:]
    assert torch.allclose(sim_output, glm(sim_input))
    # zeros_spk has another time point of nans because conv_kwargs isn't set
    sim_output = jax_to_torch(simulations[f"output_stim_spk_{i}"], 2)[..., 20:]
    assert torch.allclose(sim_output, glm_spk(sim_input)[..., 1:])
