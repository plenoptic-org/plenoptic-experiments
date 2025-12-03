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
    INIT_PATH = pathlib.Path('/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_relative_l2_vs_matlab/init_images')
    init_img = po.load_images(INIT_PATH / f"{init_img}.png")

    l2_norm = {}
    rep = model.convert_to_dict(model(img))
    for k, v in rep.items():
        l2_norm[k] = torch.linalg.vector_norm(v[~v.isnan()], ord=2)
        l2_norm[k] = torch.ones_like(v) / l2_norm[k]
        if k == "pixel_statistics":
            l2_norm[k][..., -2:] = 0
    l2_norm = model.convert_to_tensor(l2_norm)

    def loss(x, y):
        return po.tools.l2_norm(x * l2_norm, y * l2_norm)

    met = po.synth.Metamer(
        img,
        model,
        loss_function=loss,
    )
    if optimizer == "Adam":
        met.setup(init_img)
    else:
        opt_kwargs = {"max_iter": max_iter, "max_eval": max_iter,
                      "history_size": history_size, "line_search_fn": "strong_wolfe",
                      "lr": 1}
        met.setup(init_img, optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
    init_stop = time.time()
    met.synthesize(max_iter=synth_max_iter, stop_criterion=1e-16, stop_iters_to_check=1000)
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
    parser.add_argument("--max_iter", default=10, type=int)
    parser.add_argument("--history_size", default=100, type=int)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", default=200, type=int)
    parser.add_argument("--output_path", '-f', default="result.pt")
    args = vars(parser.parse_args())
    main(**args)
