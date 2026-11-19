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


from maxent_texture import load_pretrained
import numpy as np
TEXTURE_DIR = Path("/mnt/ceph/users/xzhao/Datasets")

class MaxEntTexture(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = load_pretrained()

    def forward(self, *args, **kwargs):
        return self.model.stat_model(*args, **kwargs)

def _prep(img):
    dim = min(img.shape[-2:])
    img = po.process.blur_downsample(img, int(np.floor(np.log2(dim//128))))
    img = po.process.center_crop(img, 128)
    return po.process.rescale(img, -1, 1)

def _straw():
    # straw but not the one in figure 2
    path = TEXTURE_DIR / "dtd/images/matted/matted_0085.jpg"
    return _prep(po.load_images(path, as_gray=False))

def _cracked():
    path = TEXTURE_DIR / "dtd/images/cracked/cracked_0077.jpg"
    return _prep(po.load_images(path, as_gray=False))

class InterpolationLoss(torch.nn.Module):
    def __init__(self, percent):
        super().__init__()
        model = MaxEntTexture().eval()
        img = torch.cat([_straw(), _cracked()])
        weights = torch.as_tensor([percent, 1-percent])
        print(f"Weighted sum using {weights=}")
        target_rep = (weights.unsqueeze(1) * model(img)).sum(0)
        self.register_buffer("target_rep", target_rep)
        self.model = model

    def forward(self, x):
        return po.loss.l2_norm(self.target_rep, self.model(x)).pow(2)

def loss_20_percent():
    loss = InterpolationLoss(.2)
    po.remove_grad(loss)
    return loss

def loss_40_percent():
    loss = InterpolationLoss(.4)
    po.remove_grad(loss)
    return loss

def loss_60_percent():
    loss = InterpolationLoss(.6)
    po.remove_grad(loss)
    return loss

def loss_80_percent():
    loss = InterpolationLoss(.8)
    po.remove_grad(loss)
    return loss
def status_plot(met):
    kwargs = {
        "included_plots": ["synthesis_imshow", "synthesis_loss"],
        "synthesis_imshow_kwargs": {"process_image": po.process.rescale},
        "synthesis_loss_kwargs": {"plot_penalties": True},
    }
    return po.plot.synthesis_status(met, **kwargs)
@app.default
def main(
    loss_function: Literal["loss_20_percent", "loss_40_percent", "loss_60_percent", "loss_80_percent", ],
    penalty_function: Literal["penalize_range", ],
    initial_image: Literal["rescale", ] | Path,
    optimizer_kwargs: Annotated[str, Parameter(converter=eval_ast)],
    penalty_lambda: float = 0.1,
    optimizer: Literal["Adam", ] = "Adam",
    scheduler: Literal["None", ] = "None",
    scheduler_kwargs: Annotated[str, Parameter(converter=eval_ast)] = {},
    max_iter: int = 100,
    stop_criterion: float = 0,
    stop_iters_to_check: int = 50,
    seed: int = 0,
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
    
    if loss_function == "loss_20_percent":
        loss_function = loss_20_percent()
    elif loss_function == "loss_40_percent":
        loss_function = loss_40_percent()
    elif loss_function == "loss_60_percent":
        loss_function = loss_60_percent()
    elif loss_function == "loss_80_percent":
        loss_function = loss_80_percent()
    
    if penalty_function == "penalize_range":
        penalty_function = lambda x: po.regularize.penalize_range(x, (-1, 1))
    
    if initial_image == "rescale":
        initial_image = po.process.rescale(torch.rand(1, 3, 128, 128, dtype=torch.float64, device=0), -1, 1)
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
        
    )
    stop = time.time()
    synth.save(out_path)
    
    img_path = out_path.with_suffix(".png")
    imshow_kwargs = {'process_image': po.process.rescale}
    po.plot.synthesis_imshow(synth, **imshow_kwargs).figure.savefig(img_path)
    
    
    plot_path = out_path.with_name(f"{out_path.stem}-status_plot.svg")
    status_plot(synth).savefig(plot_path)
    
    
    data.update({
        "synthesis_duration": stop - start,
        "out_path": out_path,
        "image_path": img_path,
        "synthesis_loss": synth.losses[-1].item(),
        "synthesis_penalty": synth.penalties[-1].item(),
        "device": str(device),
        "dtype": str(dtype),
    })
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(data, index=[0]).to_csv(csv_path, index=False)


if __name__ == "__main__":
    app()
