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
    coarse_to_fine: bool = True,
    seed: int = 0,
    max_iter: int = 20,
    max_eval: int | None = None,
    history_size: int = 100,
    line_search_fn: Literal[None, "strong_wolfe"] = None,
    device: str | int = 0,
    init_reduced: bool = True,
    synth_max_iter: int = 200,
    store_progress: int | None = None,
    lr: float = 0.01,
    output_dir: str = "."
):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    fn =  img
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
    po.tools.set_seed(seed)
    if coarse_to_fine:
        met = po.synth.MetamerCTF(
            img,
            model,
            loss_function=po.tools.optim.l2_norm,
            coarse_to_fine="together",
        )
        synth_kwargs = {"change_scale_criterion": None, "ctf_iters_to_check": 7}
    else:
        met = po.synth.Metamer(
            img,
            model,
            loss_function=po.tools.optim.l2_norm,
        )
        synth_kwargs = {}
    if init_reduced:
        init_img = ((torch.rand_like(img) - 0.5) * 0.1 + img.mean()).clip(min=0, max=1)
    else:
        init_img = None
    opt_kwargs = {"max_iter": max_iter, "max_eval": max_eval,
                  "history_size": history_size, "line_search_fn": line_search_fn,
                  "lr": lr}
    met.setup(init_img, optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
    start = time.time()
    met.synthesize(max_iter=synth_max_iter, store_progress=store_progress, **synth_kwargs)
    stop = time.time()
    met.save(output_dir / "metamer.pt")
    if store_progress:
        for i in range(len(met.saved_metamer)):
            fig, _ = po.synth.metamer.plot_synthesis_status(met, ylim=False, iteration=i)
            fig.savefig(output_dir / f"metamer_iter-{i}.svg")
            plt.close(fig)
    fig, _ = po.synth.metamer.plot_synthesis_status(met, ylim=False)
    fig.savefig(output_dir / f"metamer.svg")
    plt.close(fig)
    iio.imwrite(output_dir / "metamer.png", po.tools.data.convert_float_to_int(po.to_numpy(met.metamer.clip(0, 1)).squeeze()))
    data = {"filename": fn, "coarse_to_fine": coarse_to_fine, "seed": seed, "search_func": line_search_fn,
            "init_reduced": init_reduced, "LBFGS_max_iter": max_iter, "LBFGS_max_eval": max_eval,
            "LBFGS_history_size": history_size, "learning_rate": lr, "synth_iter": len(met.losses),
            "loss": met.losses[-1].item(), "synth_duration": stop-start}
    df = pd.DataFrame(data, index=[0])
    df.to_csv(output_dir / "loss.csv", index=False)
    torch.save(met.model(met.metamer), output_dir / "rep.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand LBFGS",
    )
    parser.add_argument("--img", "-i", default="fig4a")
    parser.add_argument("--coarse_to_fine", action="store_true")
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--max_iter", default=20, type=int)
    parser.add_argument("--max_eval", default=None)
    parser.add_argument("--history_size", default=100, type=int)
    parser.add_argument("--line_search_fn", default=None,)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--init_reduced", action="store_true")
    parser.add_argument("--synth_max_iter", default=200, type=int)
    parser.add_argument("--store_progress", default=None)
    parser.add_argument("--lr", default=.01, type=float)
    parser.add_argument("--output_dir", '-o', default=".")
    args = vars(parser.parse_args())
    for k in ["max_eval", "store_progress"]:
        try:
            args[k] = int(args[k])
        except:
            # then this is None
            args[k] = None
    if args["line_search_fn"] == "None":
        args["line_search_fn"] = None
    main(**args)
