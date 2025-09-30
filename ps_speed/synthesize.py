import time
import plenoptic as po
import matplotlib.pyplot as plt
from plenoptic.data.fetch import fetch_data
import torch
import argparse
import pathlib
from typing import Literal
import imageio.v3 as iio
import pandas as pd

def main(
    img: str = "fig4a",
    init_img: str = "seed-0",
    optimizer: Literal["Adam", "LBFGS"] = "Adam",
    weighted: int = 0,
    max_iter: int = 20,
    history_size: int = 100,
    device: str | int = 0,
    synth_max_iter: int = 200,
    output_path: str = "result.pt",
):
    print(po)
    torch.set_num_threads(1)
    start = time.time()
    IMG_PATH = fetch_data("portilla_simoncelli_images.tar.gz")
    try:
        device = int(device)
    except:
        pass
    device = torch.device(device)
    n_scales = 3 if img == "fig12b" else 4
    try:
        img = po.load_images(IMG_PATH / f"{img}.jpg")
    except FileNotFoundError:
        img = po.load_images(IMG_PATH / f"{img}.png")
    img = img.to(device).to(torch.float64)
    model = po.simul.PortillaSimoncelli(img.shape[-2:], n_scales=n_scales)
    model.to(device)
    INIT_PATH = pathlib.Path('/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_speed/init_images/')
    init_img = po.load_images(INIT_PATH / f"{init_img}.png")
    loss = po.tools.optim.l2_norm
    if weighted:
        weights = model.convert_to_dict(torch.ones_like(model(img)))
        for k in ["var_highpass_residual"]:
            weights[k] = weighted * torch.ones_like(weights[k])
        weights = model.convert_to_tensor(weights)
        def loss(x, y):
            return po.tools.optim.l2_norm(weights*x, weights*y)
    if optimizer == "Adam":
        met = po.synth.MetamerCTF(
            img,
            model,
            loss_function=loss,
            coarse_to_fine="together",
        )
        synth_kwargs = {"change_scale_criterion": None, "ctf_iters_to_check": 7}
        met.setup(init_img)
    else:
        met = po.synth.Metamer(
            img,
            model,
            loss_function=loss,
        )
        synth_kwargs = {}
        opt_kwargs = {"max_iter": max_iter, "max_eval": max_iter,
                      "history_size": history_size, "line_search_fn": "strong_wolfe",
                      "lr": 1}
        met.setup(init_img, optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
    init_stop = time.time()
    met.synthesize(max_iter=synth_max_iter, stop_criterion=1e-16, stop_iters_to_check=1000,
                   **synth_kwargs)
    synth_stop = time.time()
    save_dict = {"initial_time": init_stop - start,
                 "synth_time": synth_stop - start,
                 "metamer": met.metamer}
    torch.save(save_dict, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand LBFGS",
    )
    parser.add_argument("--img", "-i", default="fig4a")
    parser.add_argument("--init_img", "-t", default="seed-0")
    parser.add_argument("--optimizer", "-o", default="Adam")
    parser.add_argument("--weighted", default=0, type=float)
    parser.add_argument("--max_iter", default=10, type=int)
    parser.add_argument("--history_size", default=100, type=int)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", default=200, type=int)
    parser.add_argument("--output_path", '-f', default="result.pt")
    args = vars(parser.parse_args())
    main(**args)
