import ast
import time
from cyclopts import App, Parameter, Token
from typing import Literal, Annotated, Sequence
import plenoptic as po
import matplotlib.pyplot as plt
import torch
import pandas as pd
from pathlib import Path
import warnings

app = App()

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "figure.dpi": 72,
})

def eval_torch(type_, tokens: Sequence[Token]):
    # converts torch/None strings to appropriate objects
    value = tokens[0].value
    if isinstance(value, str):
        if value.startswith("torch."):
            return eval(value)
    else:
        return value


def eval_ast(type_, tokens: Sequence[Token]):
    return ast.literal_eval(tokens[0].value)


def load_image(path: Path):
    if path.suffix == ".pt":
        loaded = torch.load(path, weights_only=True)
        return loaded["_genericsynthesis"]
    else:
        return po.load_images(path)


class MEI(torch.nn.Module):
    def __init__(self, pad_mode, minmax, model="gaussian"):
        super().__init__()
        if model == "gaussian":
            self.model = po.models.Gaussian(30, pad_mode=pad_mode, cache_filt=True)
        elif model == "centersurround":
            self.model = po.models.CenterSurround(30, pad_mode=pad_mode, cache_filt=True)
        elif model == "LGC":
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, pretrained=True, cache_filt=True)
        elif model == "LGC-big":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] *= 2
            state_dict["center_surround.surround_std"] *= 2
            self.model.load_state_dict(state_dict)
        elif model == "LGC-small":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] /= 2
            state_dict["center_surround.surround_std"] /= 2
            self.model.load_state_dict(state_dict)
        elif model == "LGC-bigish":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] *= 1.5
            state_dict["center_surround.surround_std"] *= 1.5
            self.model.load_state_dict(state_dict)
        elif model == "LGC-smallish":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] /= 1.5
            state_dict["center_surround.surround_std"] /= 1.5
            self.model.load_state_dict(state_dict)
        self.minmax = minmax
        if minmax == "expmax":
            self.process = lambda x: torch.exp(-x)
        elif minmax == "max":
            self.process = lambda x: -x
        self.eval()
        po.remove_grad(self)

    def forward(self, x):
        return self.process(self.model(x).mean())


# def mei_gaussian_circular_expmax():
#     return MEI("circular", "expmax", "gaussian")

# def mei_gaussian_constant_expmax():
#     return MEI("constant", "expmax", "gaussian")

# def mei_gaussian_circular_max():
#     return MEI("circular", "max", "gaussian")

# def mei_gaussian_constant_max():
#     return MEI("constant", "max", "gaussian")

# def mei_centersurround_circular_expmax():
#     return MEI("circular", "expmax", "centersurround")

# def mei_centersurround_constant_expmax():
#     return MEI("constant", "expmax", "centersurround")

# def mei_centersurround_circular_max():
#     return MEI("circular", "max", "centersurround")

# def mei_centersurround_constant_max():
#     return MEI("constant", "max", "centersurround")

# def mei_LGC_circular_expmax():
#     return MEI("circular", "expmax", "LGC")

# def mei_LGC_constant_expmax():
#     return MEI("constant", "expmax", "LGC")

def mei_LGC_circular_max():
    return MEI("circular", "max", "LGC")

def mei_LGC_constant_max():
    return MEI("constant", "max", "LGC")

def mei_LGCbig_circular_max():
    return MEI("circular", "max", "LGC-big")

def mei_LGCbig_constant_max():
    return MEI("constant", "max", "LGC-big")

def mei_LGCbigish_circular_max():
    return MEI("circular", "max", "LGC-bigish")

def mei_LGCbigish_constant_max():
    return MEI("constant", "max", "LGC-bigish")

def mei_LGCsmallish_circular_max():
    return MEI("circular", "max", "LGC-smallish")

def mei_LGCsmallish_constant_max():
    return MEI("constant", "max", "LGC-smallish")

def mei_LGCsmall_circular_max():
    return MEI("circular", "max", "LGC-small")

def mei_LGCsmall_constant_max():
    return MEI("constant", "max", "LGC-small")
# def zero_mean():
#     def penalty(x):
#         mean_penalty = x.mean().pow(2)
#         range_penalty = po.regularize.penalize_range(x, (-1, 1))
#         return mean_penalty + range_penalty
#     return penalty

def just_range():
    return po.regularize.penalize_range
def rand():
    return torch.rand((1, 1, 128, 128))

# def rand_small():
#     return (0.1 * torch.rand((1, 1, 128, 128))) + 0.5

# def randn():
#     return 0.1 * torch.randn((1, 1, 128, 128))

def randn_ctr():
    return (0.1 * torch.randn((1, 1, 128, 128))) + 0.5
def status_plot(synth):
    kwargs = {
        "included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"],
        "synthesis_loss_kwargs": {"plot_penalties": True},
    }
    fig = po.plot.synthesis_status(synth, **kwargs)
    if synth.loss_function.minmax == "max":
        fig.axes[1].set(yscale="linear")
    return fig
@app.default
def main(
    loss_function: Literal["mei_LGC_circular_max", "mei_LGC_constant_max", "mei_LGCbig_circular_max", "mei_LGCbig_constant_max", "mei_LGCbigish_circular_max", "mei_LGCbigish_constant_max", "mei_LGCsmallish_circular_max", "mei_LGCsmallish_constant_max", "mei_LGCsmall_circular_max", "mei_LGCsmall_constant_max", ],
    penalty_function: Literal["just_range", ],
    penalty_lambda: float,
    initial_image: Literal["rand", "randn_ctr", ] | Path,
    optimizer: Literal["Adam", ],
    optimizer_kwargs: Annotated[str, Parameter(converter=eval_ast)],
    scheduler: Literal["None", ] = "None",
    scheduler_kwargs: Annotated[str, Parameter(converter=eval_ast)] = {},
    max_iter: int = 100,
    stop_criterion: float = 0,
    stop_iters_to_check: int = 50,
    seed: int = 0,n_frames: int = 50,
    out_path: Path = Path("synth.pt"),
    device: int | str = "cpu",
    dtype: Annotated[str, Parameter(converter=eval_torch)] = torch.float32,
):
    """Synthesize some GenericSynthesis images."""
    data = {
        "loss_function": loss_function,
        "penalty_function": penalty_function,
        "initial_image": initial_image,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "penalty_lambda": penalty_lambda,
        "max_iter": max_iter,
        "stop_criterion": stop_criterion,
        "stop_iters_to_check": stop_iters_to_check,
        "seed": seed,
    }
    if optimizer_kwargs:
        data.update({f"optimizer_kwargs_{k}": v for k, v in optimizer_kwargs.items()})
    if scheduler_kwargs:
        data.update({f"scheduler_kwargs_{k}": v for k, v in scheduler_kwargs.items()})
    
    if loss_function == "mei_LGC_circular_max":
        loss_function = mei_LGC_circular_max()
    elif loss_function == "mei_LGC_constant_max":
        loss_function = mei_LGC_constant_max()
    elif loss_function == "mei_LGCbig_circular_max":
        loss_function = mei_LGCbig_circular_max()
    elif loss_function == "mei_LGCbig_constant_max":
        loss_function = mei_LGCbig_constant_max()
    elif loss_function == "mei_LGCbigish_circular_max":
        loss_function = mei_LGCbigish_circular_max()
    elif loss_function == "mei_LGCbigish_constant_max":
        loss_function = mei_LGCbigish_constant_max()
    elif loss_function == "mei_LGCsmallish_circular_max":
        loss_function = mei_LGCsmallish_circular_max()
    elif loss_function == "mei_LGCsmallish_constant_max":
        loss_function = mei_LGCsmallish_constant_max()
    elif loss_function == "mei_LGCsmall_circular_max":
        loss_function = mei_LGCsmall_circular_max()
    elif loss_function == "mei_LGCsmall_constant_max":
        loss_function = mei_LGCsmall_constant_max()
    
    if penalty_function == "just_range":
        penalty_function = just_range()
    
    if initial_image == "rand":
        initial_image = rand()
    elif initial_image == "randn_ctr":
        initial_image = randn_ctr()
    else:
        initial_image = load_image(initial_image)
    
    if optimizer == "Adam":
        optimizer = torch.optim.Adam
    
    if scheduler == "None":
        scheduler = None
    
    synth = po.GenericSynthesis(
        loss_function=loss_function,
        penalty_function=penalty_function,
        penalty_lambda=penalty_lambda,
    )
    device = torch.device(device)
    synth.to(device)
    synth.to(dtype)
    
    synth.setup(
        initial_image=initial_image,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
    )
    
    print(synth)
    
    start = time.time()
    synth.synthesize(
        max_iter=max_iter,
        stop_criterion=stop_criterion,
        stop_iters_to_check=stop_iters_to_check,
        store_progress=max(max_iter//n_frames, 1),
    )
    stop = time.time()
    synth.save(out_path)
    
    ani_path = out_path.with_suffix(".mp4")
    animate_kwargs = {"included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"], "synthesis_loss_kwargs": {"plot_penalties": True}}
    po.plot.synthesis_animate(synth, **animate_kwargs).save(ani_path)
    
    img_path = out_path.with_suffix(".png")
    imshow_kwargs = {}
    po.plot.synthesis_imshow(synth, **imshow_kwargs).figure.savefig(img_path)
    
    
    plot_path = out_path.with_name(f"{out_path.stem}-status_plot.svg")
    status_plot(synth).savefig(plot_path)
    
    
    data.update({
        "synthesis_duration": stop - start,
        "out_path": out_path,
        "image_path": img_path,"animate_path": ani_path,
        "synthesis_loss": synth.losses[-1].item(),
        "synthesis_penalty": synth.penalties[-1].item(),
        "device": str(device),
        "dtype": str(dtype),
    })
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(data, index=[0]).to_csv(csv_path, index=False)


if __name__ == "__main__":
    app()
