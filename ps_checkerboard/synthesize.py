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
    seed: int = 0,
    max_iter: int = 20,
    max_eval: int | None = None,
    history_size: int = 100,
    line_search_fn: Literal[None, "strong_wolfe"] = None,
    device: str | int = 0,
    synth_max_iter: int = 200,
    lr: float = 0.01,
    output_dir: str = "."
):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    try:
        device = int(device)
    except:
        pass
    device = torch.device(device)
    img = po.load_images("checkerboard.jpg")
    img = img.to(device).to(torch.float64)
    model = po.simul.PortillaSimoncelli(img.shape[-2:], spatial_corr_width=7)
    model.to(device)
    po.tools.set_seed(seed)
    weights = model.convert_to_dict(torch.ones_like(model(image)))
    if "pixel_statistics" in weights:
        # reweight the pixel min/max and the variance of the highpass residuals, since
        # they're weird.
        weights["pixel_statistics"][..., -2:] = minmax_weight
    k = "var_highpass_residual"
    if k in weights:
        weights[k] = highpass_weight * torch.ones_like(weights[k])
    weights = model.convert_to_tensor(weights)
    nan_mask = model(img).isnan()

    def loss(x: Tensor, y: Tensor) -> Tensor:  # numpydoc ignore=GL08
        x[nan_mask] = 0
        y[nan_mask] = 0
        return l2_norm(weights * x, weights * y)

    met = po.synth.Metamer(img, model, loss_function=los,)
    opt_kwargs = {"max_iter": max_iter, "max_eval": max_eval,
                  "history_size": history_size, "line_search_fn": line_search_fn,
                  "lr": lr}
    met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
    start = time.time()
    met.synthesize(max_iter=synth_max_iter, **synth_kwargs)
    stop = time.time()
    met.save(output_dir / "metamer.pt")
    width_ratios = {"plot_representation_error": 3.1}
    fig, _ = po.synth.metamer.plot_synthesis_status(met, ylim=False, width_ratios=width_ratios)
    fig.savefig(output_dir / f"metamer.svg")
    plt.close(fig)
    iio.imwrite(output_dir / "metamer.png", po.tools.data.convert_float_to_int(po.to_numpy(met.metamer.clip(0, 1)).squeeze()))
    data = {"coarse_to_fine": coarse_to_fine, "seed": seed, "search_func": line_search_fn,
            "init_reduced": init_reduced, "LBFGS_max_iter": max_iter, "LBFGS_max_eval": max_eval,
            "LBFGS_history_size": history_size, "learning_rate": lr, "synth_iter": len(met.losses),
            "loss": met.losses[-1].item(), "synth_duration": stop-start}
    df = pd.DataFrame(data, index=[0])
    df.to_csv(output_dir / "loss.csv", index=False)
    torch.save(met.model(met.metamer), output_dir / "rep.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand checkerboard synth",
    )
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--max_iter", default=20, type=int)
    parser.add_argument("--max_eval", default=None)
    parser.add_argument("--history_size", default=100, type=int)
    parser.add_argument("--line_search_fn", default=None,)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", default=200, type=int)
    parser.add_argument("--lr", default=.01, type=float)
    parser.add_argument("--output_dir", '-o', default=".")
    args = vars(parser.parse_args())
    try:
        args["max_eval"] = int(args["max_eval"])
    except:
        # then this is None
        args["max_eval"] = None
    if args["line_search_fn"] == "None":
        args["line_search_fn"] = None
    main(**args)
