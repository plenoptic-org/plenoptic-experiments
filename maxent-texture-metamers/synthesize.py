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
        return loaded["_metamer"]
    else:
        return po.load_images(path)


#!/usr/bin/env python3

TEXTURE_DIR = Path("/mnt/ceph/users/xzhao/Datasets")

def _prep(img):
    dim = min(img.shape[-2:])
    img = po.process.blur_downsample(img, dim//128)
    img = po.process.center_crop(img, 128)
    return po.process.rescale(img, -1, 1)

def brick():
    # brick but not the one in figure 2
    path = TEXTURE_DIR / "cropped1024-selected/10D_9452.tif"
    return _prep(po.load_images(path))

def straw():
    # straw but not the one in figure 2
    path = TEXTURE_DIR / "dtd/images/matted/matted_0085.jpg"
    return _prep(po.load_images(path))

def cracked():
    path = TEXTURE_DIR / "dtd/images/cracked/cracked_0077.jpg"
    return _prep(po.load_images(path))
from maxent_texture import load_pretrained

class MaxEntTexture(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = load_pretrained()

    def forward(self, *args, **kwargs):
        return self.model.stat_model(*args, **kwargs)

def maxent_texture():
    return MaxEntTexture()
@app.default
def main(
    image: Literal["brick", "straw", "cracked", ] | Path,
    optimizer_kwargs: Annotated[str, Parameter(converter=eval_ast)],
    model: Literal["maxent_texture", ] = "maxent_texture",
    loss_function: Literal["mse", ] = "mse",
    penalty_function: Literal["penalize_range", ] = "penalize_range",
    penalty_lambda: float = 0.1,
    initial_image: Literal["rand_like", ] | Path = "rand_like",
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
    """Synthesize some Metamer images."""
    data = {
        "image": image,
        "model": model,
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
    
    if image == "brick":
        image = brick()
    elif image == "straw":
        image = straw()
    elif image == "cracked":
        image = cracked()
    else:
        image = load_image(image)
    
    if model == "maxent_texture":
        model = maxent_texture()
    
    if loss_function == "mse":
        loss_function = po.metric.mse
    
    if penalty_function == "penalize_range":
        penalty_function = po.regularize.penalize_range
    
    if initial_image == "rand_like":
        initial_image = torch.rand_like(image)
    else:
        initial_image = load_image(initial_image)
    
    if optimizer == "Adam":
        optimizer = torch.optim.Adam
    
    if scheduler == "None":
        scheduler = None
    
    po.remove_grad(model)
    
    synth = po.Metamer(
        image=image,
        model=model,
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
        store_progress=max_iter//n_frames,
    )
    stop = time.time()
    synth.save(out_path)
    
    ani_path = out_path.with_suffix(".mp4")
    po.plot.synthesis_animate(synth).save(ani_path)
    
    img_path = out_path.with_suffix(".png")
    po.plot.synthesis_imshow(synth).figure.savefig(img_path)
    
    plot_path = out_path.with_suffix(".svg")
    po.plot.synthesis_status(synth).savefig(plot_path)
    
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
