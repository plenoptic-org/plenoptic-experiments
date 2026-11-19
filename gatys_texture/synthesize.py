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


import numpy as np
TEXTURE_DIR = Path("/mnt/ceph/users/xzhao/Datasets")

def _prep(img):
    dim = min(img.shape[-2:])
    n_downsample = int(np.floor(np.log2(dim//224)))
    if n_downsample > 0:
        img = po.process.blur_downsample(img, n_downsample)
    return po.process.center_crop(img, 224)

def brick():
    # brick but not the one in figure 2
    path = TEXTURE_DIR / "cropped1024-selected/10D_9452.tif"
    return _prep(po.load_images(path, as_gray=False))

def straw():
    # straw but not the one in figure 2
    path = TEXTURE_DIR / "dtd/images/matted/matted_0085.jpg"
    return _prep(po.load_images(path, as_gray=False))

def cracked():
    path = TEXTURE_DIR / "dtd/images/cracked/cracked_0077.jpg"
    return _prep(po.load_images(path, as_gray=False))
import torchvision
import einops

def gatys():
    # gatys actually use a modified version of VGG19, see
    # https://github.com/honzukka/texture-synthesis-pytorch/
    weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
    tv_model = torchvision.models.vgg19(weights=weights).eval()
    # This model's transform consists of resizing, cropping, and normalizing.
    # We recommend only including the normalizing in the transform.
    tv_transform = weights.transforms()
    norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
    layers = ["features.1", "features.4", "features.9", "features.18", "features.27"]
    def gram(x):
        return {
            k: einops.einsum(v, v, "b c1 h w, b c2 h w -> b c1 c2")
            for k, v in x.items()
        }
    return po.models.DeepNetFeatures(tv_model, layers, norm, gram)
def l2_norm_squared():
    return lambda x, y: po.loss.l2_norm(x, y).pow(2)
# def randn_rescale(image):
#     init_img = 0.01 * (torch.rand_like(image) - 0.5)
#     return po.process.rescale(init_img, 0, 1)

def randn(image):
    init_img = 0.01 * torch.randn_like(image)
    return 0.5 + init_img

def rand(image):
    init_img = torch.rand_like(image)
    return init_img

def rand_small(image):
    init_img = 0.1 * torch.rand_like(image)
    return 0.5 + init_img
def status_plot(met):
    kwargs = {
        "included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"],
        "synthesis_loss_kwargs": {"plot_penalties": True},
    }
    return po.plot.synthesis_status(met, **kwargs)
@app.default
def main(
    image: Literal["brick", "straw", "cracked", ] | Path,
    model: Literal["gatys", ],
    loss_function: Literal["l2_norm_squared", ],
    penalty_function: Literal["penalize_range", ],
    penalty_lambda: float,
    initial_image: Literal["randn", "rand", "rand_small", ] | Path,
    optimizer: Literal["LBFGS", ],
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
    
    if model == "gatys":
        model = gatys()
    
    if loss_function == "l2_norm_squared":
        loss_function = l2_norm_squared()
    
    if penalty_function == "penalize_range":
        penalty_function = po.regularize.penalize_range
    
    if initial_image == "randn":
        initial_image = randn(image)
    elif initial_image == "rand":
        initial_image = rand(image)
    elif initial_image == "rand_small":
        initial_image = rand_small(image)
    else:
        initial_image = load_image(initial_image)
    
    if optimizer == "LBFGS":
        optimizer = torch.optim.LBFGS
    
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
