#!/usr/bin/env python3

import plenoptic as po
import torch
import typer
from pathlib import Path
from typing import Annotated, Literal
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

app = typer.Typer(add_completion=False)


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
        if link_func == "jax.numpy.exp":
            self.link_func = torch.exp
        else:
            raise ValueError(f"Don't know how to handle {link_func=}")
        self.load_state_dict(state_dict)

    def forward(self, x, **kwargs):
        return self.link_func(self.conv(x, **kwargs))

    @classmethod
    def load_nemos_glm(cls, path):
        # nemos convention is reverse of torch's
        coeffs_npz = np.load(path)
        try:
            # this is a simple GLM
            weight = jax_to_torch(coeffs_npz["item::strkey:coef_"][::-1])
        except KeyError:
            # this is a GLM that was fit using a pytree, specifying the stimulus filter
            weight = jax_to_torch(coeffs_npz["dict::strkey:coef_::item::strkey:stim"][::-1])
        bias = jax_to_torch(coeffs_npz["item::strkey:intercept_"])
        link_func = coeffs_npz["item::strkey:inverse_link_function"]
        return cls(weight=weight, bias=bias, link_func=link_func)


def plot_met(mets, labels, save_path="tmp.svg"):
    if not hasattr(mets, "__len__"):
        mets = [mets]
    if not hasattr(labels, "__len__"):
        labels = [labels]
    gs = mpl.gridspec.GridSpec(4, 2, width_ratios=[1, 3])
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    ax = fig.add_subplot(gs[1:3, 0])
    met = mets[0]
    ax.plot(po.to_numpy(met.model.conv.weight.squeeze()))
    ax.set_title("Filter")
    ax = fig.add_subplot(gs[:2, 1])
    ax.set_title("Stimuli")
    ax.plot(met.image.squeeze(), label="Real stimulus")
    for met, label in zip(mets, labels):
        ax.plot(po.to_numpy(met.metamer.squeeze()), "--", label=label)
    ax.legend()
    n_timepts = met.image.shape[-1]
    ax.set(xlim=(0, n_timepts))
    model_stim = met.model(met.image).squeeze()
    init_x = n_timepts - len(model_stim)
    x = np.arange(init_x, met.image.shape[-1])
    ax = fig.add_subplot(gs[2:, 1])
    ax.set_title("Model response")
    ax.plot(x, po.to_numpy(model_stim), label="Real stimulus")
    for met, label in zip(mets, labels):
        ax.plot(x, po.to_numpy(met.model(met.metamer).squeeze()), "--", label=label)
    ax.legend()
    ax.set(xlim=(0, n_timepts))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def prepare_met(stim, model, penalty=None):
    po.remove_grad(model)
    model.eval()

    if penalty is None:
        met = po.Metamer(stim, model, penalty_lambda=0)
    elif penalty == "range":
        met = po.Metamer(stim, model, penalty_function=lambda x: po.regularize.penalize_range(x, (-.5, .5)))
    else:
        met = po.Metamer(stim, model, penalty_function=penalty)

    met.setup(optimizer=torch.optim.LBFGS)
    return met


def validate_model(model, simulation_key, simulations="nemos_simulations.npz"):
    if isinstance(simulations, str):
        simulations = np.load("nemos_simulations.npz")

    for i in range(simulations["n_simulations"]):
        sim_input = jax_to_torch(simulations[f"input_{i}"], 2)
        sim_output = jax_to_torch(simulations[f"output_{simulation_key}_{i}"], 2)[..., 19:]
        assert torch.allclose(sim_output, glm(sim_input))
        # zeros_spk has another time point of nans because conv_kwargs isn't set
        sim_output = jax_to_torch(simulations[f"output_{simulation_key}_{i}"], 2)[..., 20:]
        assert torch.allclose(sim_output, glm_spk(sim_input)[..., 1:])


def prepare_penalty(stim, penalty):
    # this remaps so that the minumum is at target, at which point are function is 0. this
    # works as long as we have a finite target (if our target was -inf, it wouldn't)
    def remap(x, target=0):
        return (x-target).pow(2)

    def corr_penalty(metamer, target=0):
        penalty = torch.corrcoef(torch.stack([stim.squeeze(), metamer.squeeze()], 0))[0, 1]
        return remap(penalty, target)

    if penalty == "corr":
        penalty = lambda x: corr_penalty(x, 1) + po.regularize.penalize_range(x, (-.5, .5))
    elif penalty == "uncorr":
        penalty = lambda x: corr_penalty(x, 0) + po.regularize.penalize_range(x, (-.5, .5))
    elif penalty == "anticorr":
        penalty = lambda x: corr_penalty(x, -1) + po.regularize.penalize_range(x, (-.5, .5))
    elif penalty == "mse":
        penalty = lambda x: (x.squeeze() - stim.squeeze()).pow(2).mean()

    return penalty


@app.command()
def synthesize(
    glm_path: Annotated[Path, typer.Argument(help="Path to npz file created by nemos's save_params().")],
    penalty: Annotated[Literal[None, "range", "corr", "uncorr", "anticorr", "mse"], typer.Option(help="Penalty to use.")] = None,
    save_stem: Annotated[str, typer.Option(help="Stem to use for saving metamer")] = "met",
    stimulus_path: Annotated[Path, typer.Argument(help="Path to npz file containing the stimulus.")] = "nemos_stimulus.npz",
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 1,
):
    """Synthesize. """
    po.set_seed(seed)
    glm = GLM.load_nemos_glm(glm_path)

    # do plenoptic
    stim = jax_to_torch(np.load(stimulus_path)["stimulus"], 2)[..., :200]
    penalty = prepare_penalty(stim, penalty)

    met = prepare_met(stim, glm, penalty)
    met.synthesize(1000, stop_criterion=1e-20)
    met.save(f"{save_stem}.pt")


@app.command()
def plot(
    glm_path: Annotated[Path, typer.Argument(help="Path to npz file created by nemos's save_params().")],
    penalty: Annotated[Literal[None, "range", "corr", "uncorr", "anticorr", "mse"], typer.Option(help="Penalty to use.")] = None,
    load_stem: Annotated[str, typer.Option(help="Stem to use for loading metamer")] = "met",
    metamer_label: Annotated[str, typer.Option(help="Label to use in legend")] = "Metamer",
):
    """Plot synthesized metamers."""

    glm = GLM.load_nemos_glm(glm_path)
    # do plenoptic
    stim = jax_to_torch(np.load(stimulus_path)["stimulus"], 2)[..., :200]
    penalty = prepare_penalty(stim, penalty)
    met = prepare_met(stim, glm, penalty)
    met.load(f"{load_stem}.pt")
    plot_met(met, metamer_label, f"{load_stem}.svg")


if __name__ == "__main__":
    app()
