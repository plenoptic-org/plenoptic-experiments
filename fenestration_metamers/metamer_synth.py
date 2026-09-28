import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

import fenestration as fen
import time
import pandas as pd
import ast
import pathlib
import plenoptic as po
from typing import Annotated
import typer

plt.rcParams["figure.dpi"] = 72
plt.rcParams['savefig.bbox'] = "tight"

def load_images(image_name: str, img_size: int = 200):
    if image_name == "reptile":
        img = po.data.reptile_skin()
    elif image_name == "einstein":
        img = po.data.einstein()
    img = po.process.center_crop(img, img_size)
    return img

def _convert_to_int(device):
    try:
        device = int(device)
    except ValueError:
        pass
    return device

def main(
    device: Annotated[str, typer.Option(help="Path to save figure at.", callback=_convert_to_int)] = "cpu",
    optimizer_name: Annotated[str, typer.Option(help="optimizer")] = "LBFGS",
    opt_kwargs: Annotated[str, typer.Option(help="optimizer args", callback=ast.literal_eval)] = "{}",
    scaling: Annotated[float, typer.Option(help="model scaling")] = 0.8,
    image_name: Annotated[str, typer.Option(help="image to load")] = "reptile",
    max_iter: Annotated[int, typer.Option(help="number of iterations")] = 400,
    fig_save_path: Annotated[pathlib.Path | None, typer.Option(help="Path to save figure at.")] = None,
):
    img = load_images(image_name).to(device)
    model = fen.PoolingWindows(scaling, img.shape[-2:]).to(device)
    model.eval()
    po.remove_grad(model)
    met = po.Metamer(img, model)
    optimizer = eval(f"torch.optim.{optimizer_name}")
    print(opt_kwargs)
    print(img.device)
    met.setup(optimizer=optimizer, optimizer_kwargs=opt_kwargs)
    start = time.time()
    met.synthesize(max_iter=max_iter, stop_criterion=1e-6)
    stop = time.time()
    if fig_save_path is None:
        fig_save_path = f"fenestration_metamer_image-{image_name}_scaling-{scaling}.png"
    ax = po.plot.synthesis_imshow(met)
    met_img_path = fig_save_path.with_name(f"{fig_save_path.stem}_metamer.png")
    ax.figure.savefig(met_img_path)
    fig = po.plot.synthesis_status(met)
    fig.savefig(fig_save_path)
    data = {"image_name": image_name, "model": "luminance", "optimizer": optimizer_name,
            "device": device, "status_path": fig_save_path,
            "image_path": met_img_path, "loss": met.losses[-1].item(),
            "penalty": met.penalties[-1].item(), "synth_duration": stop - start,
            }
    data.update(opt_kwargs)
    pd.DataFrame(data, index=[0]).to_csv(fig_save_path.with_suffix(".csv"), index=False)


if __name__ == '__main__':
    typer.run(main)
