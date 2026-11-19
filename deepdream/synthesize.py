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


import torchvision
import numpy as np
import torch.nn.functional as F


class ChannelModel(torch.nn.Module):
    def __init__(self, layer, channel_n, upsample_scales=False, jitter_pix=False):
        super().__init__()
        if isinstance(layer, str):
            layer = [layer]
        if len(layer) != 1:
            raise ValueError("Only works with single layer!")
        self._layer = layer[0]
        weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
        deepnet = torchvision.models.googlenet(weights=weights).eval()
        transform = weights.transforms()
        norm = [torchvision.transforms.Normalize(transform.mean, transform.std)]
        if jitter_pix:
            def inner_pad(x):
                return F.pad(x, [0, 0, int(jitter_pix * 1.5), int(jitter_pix*1.5)], mode="constant", value=0.5)
            def inner_jitter1(x):
                return torch.roll(x, (np.random.choice(jitter_pix),
                                      np.random.choice(jitter_pix)),
                                  (-2, -1))
            norm = [*norm, inner_pad, inner_jitter1]
        if upsample_scales:
            def inner_upsample(x):
                scale = np.random.choice(upsample_scales)
                upsample = torch.nn.Upsample(scale_factor=scale, mode="bilinear",
                                             align_corners=True)
                return upsample(x)
            norm = [*norm, inner_upsample]
        if jitter_pix:
            def inner_jitter2(x):
                return torch.roll(x, (np.random.choice(jitter_pix//2),
                                      np.random.choice(jitter_pix//2)),
                                  (-2, -1))
            norm = [*norm, inner_jitter2]
        norm = torchvision.transforms.Compose(norm)
        self.model = po.models.DeepNetFeatures(deepnet, layer, norm)
        self.channel_n = channel_n

    def forward(self, x):
        x = self.model.transform(x)
        out = self.model.extractor(x)
        out = {k: v[:, self.channel_n] for k, v in out.items()}
        out = self.model.convert_to_tensor(out)
        if (out < 0).any():
            raise ValueError("NEGATIVE VALUES")
        return out


class FeatureVis(torch.nn.Module):
    def __init__(self, layer, channel_n, upsample, jitter, minmax="max"):
        super().__init__()
        self.model = ChannelModel(layer, channel_n, upsample, jitter)
        if minmax == "min":
            self.process = lambda x: x
        elif minmax == "max":
            self.process = lambda x: torch.exp(-x)
        elif minmax == "realmax":
            self.process = lambda x: -x

    def forward(self, x):
        return self.process(self.model(x).mean())


# def feature_vis_min_4c_448():
#     return FeatureVis("inception4c", 448, "min")


def feature_vis_max_4c_448():
    return FeatureVis("inception4c", 448, False, False, "max")


# def feature_vis_min_4c_448_transformed():
#     return FeatureVis("inception4c", 448, 1.1, True, "min")


def feature_vis_max_4c_448_transformed():
    return FeatureVis("inception4c", 448, [0.8, 0.9, 1.1, 1.2] + 4*[1], 8, "max")


# def feature_vis_realmax_4a_476_transformed():
#     return FeatureVis("inception4a", 476, [0.8, 0.9, 1.1, 1.2] + 4*[1], 8, "realmax")

# def feature_vis_realmax_4a_476():
#     return FeatureVis("inception4a", 476, False, False, "realmax")

def feature_vis_max_4a_476_transformed():
    return FeatureVis("inception4a", 476, [0.8, 0.9, 1.1, 1.2] + 4*[1], 8, "max")

def feature_vis_max_4a_476():
    return FeatureVis("inception4a", 476, False, False, "max")
def randn():
    return 0.01 * torch.randn((1, 3, 224, 224))

def randn_add():
    return 0.5 + (0.01 * torch.randn((1, 3, 224, 224)))

def rand_small():
    return 0.5 + (0.1 * torch.rand((1, 3, 224, 224)))

def rand():
    return torch.rand((1, 3, 224, 224))
def status_plot(met):
    kwargs = {
        "included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"],
        # "synthesis_imshow_kwargs": {"process_image": po.process.rescale},
    }
    fig = po.plot.synthesis_status(met, **kwargs)
    # fig.axes[1].set(yscale="linear")
    return fig
@app.default
def main(
    loss_function: Literal["feature_vis_max_4c_448", "feature_vis_max_4c_448_transformed", "feature_vis_max_4a_476_transformed", "feature_vis_max_4a_476", ],
    penalty_lambda: float,
    optimizer_kwargs: Annotated[str, Parameter(converter=eval_ast)],
    penalty_function: Literal["penalize_range", ] = "penalize_range",
    initial_image: Literal["randn", "randn_add", "rand_small", "rand", ] | Path = "randn",
    optimizer: Literal["Adam", ] = "Adam",
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
    
    if loss_function == "feature_vis_max_4c_448":
        loss_function = feature_vis_max_4c_448()
    elif loss_function == "feature_vis_max_4c_448_transformed":
        loss_function = feature_vis_max_4c_448_transformed()
    elif loss_function == "feature_vis_max_4a_476_transformed":
        loss_function = feature_vis_max_4a_476_transformed()
    elif loss_function == "feature_vis_max_4a_476":
        loss_function = feature_vis_max_4a_476()
    
    if penalty_function == "penalize_range":
        penalty_function = po.regularize.penalize_range
    
    if initial_image == "randn":
        initial_image = randn()
    elif initial_image == "randn_add":
        initial_image = randn_add()
    elif initial_image == "rand_small":
        initial_image = rand_small()
    elif initial_image == "rand":
        initial_image = rand()
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
    animate_kwargs = {"included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"]}
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
