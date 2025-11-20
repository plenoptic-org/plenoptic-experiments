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
    img: str = "checkerboard",
    seed: int = 0,
    max_iter: int = 20,
    max_eval: int | None = None,
    history_size: int = 100,
    line_search_fn: Literal[None, "strong_wolfe"] = None,
    device: str | int = 0,
    synth_max_iter: int = 200,
    range_lmbda: float = .1,
    mn: float = 1.,
    mag_std: float = 1.,
    autocorr_recon: float = 1.,
    skew_recon: float = 1.,
    kurt_recon: float = 1.,
    cross_ori: float = 1.,
    cross_scale_mag: float = 1.,
    cross_scale_real: float = 1.,
    autocorr_mag: float = 1.,
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
    IMG_PATH = fetch_data("portilla_simoncelli_images.tar.gz")
    fn = img
    try:
        img = po.load_images(IMG_PATH / f"{img}.jpg")
    except FileNotFoundError:
        img = po.load_images(f"{img}.jpg")
    img = img.to(device).to(torch.float64)
    model = po.simul.PortillaSimoncelli(img.shape[-2:], spatial_corr_width=7)
    model.to(device)
    po.tools.set_seed(seed)
    weights = model.convert_to_dict(torch.ones_like(model(img)))
    if "pixel_statistics" in weights:
        # reweight the pixel min/max and the variance of the highpass residuals, since
        # they're weird.
        weights["pixel_statistics"][..., -2:] = 0
        weights["pixel_statistics"][..., 0] = mn
    weight_dict = {"var_highpass_residual": 100, "magnitude_std": mag_std,
                   "auto_correlation_reconstructed": autocorr_recon,
                   "skew_reconstructed": skew_recon, "kurtosis_reconstructed": kurt_recon,
                   "cross_orientation_correlation_magnitude": cross_ori,
                   "cross_scale_correlation_magnitude": cross_scale_mag,
                   "cross_scale_correlation_real": cross_scale_real,
                   "auto_correlation_magnitude": autocorr_mag}
    for k, v in weight_dict.items():
        weights[k] = v * torch.ones_like(weights[k])
    weights = model.convert_to_tensor(weights)
    nan_mask = model(img).isnan()

    def loss(x, y):
        y = y.clone()
        x[nan_mask] = 0
        y[nan_mask] = 0
        return po.tools.l2_norm(weights * x, weights * y)

    met = po.synth.Metamer(img, model, loss_function=loss, allowed_range=(img.min(), img.max()),
                           range_penalty_lambda=range_lmbda)
    synth_kwargs = {}
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
    data = {"image": fn, "seed": seed, "search_func": line_search_fn,
            "LBFGS_max_iter": max_iter, "LBFGS_max_eval": max_eval,
            "LBFGS_history_size": history_size, "learning_rate": lr, "synth_iter": len(met.losses),
            "synth_loss": met.losses[-1].item(), "synth_duration": stop-start}
    data.update(weight_dict)
    data["mean_weight"] = mn

    def mse(x, y):
        return (x-y).pow(2).nanmean().item()
    def sse(x, y):
        return (x-y).pow(2).nansum().item()
    def l2_norm(x, y):
        return (x-y).pow(2).nansum().sqrt().item()

    df = []
    rep = model(met.image)
    met_rep = model(met.metamer)
    rep_dict = model.convert_to_dict(rep)
    met_rep_dict = model.convert_to_dict(met_rep)

    for func in [mse, sse, l2_norm]:
        d = data.copy()
        d["loss"] = func(rep, met_rep)
        d["loss_type"] = "overall"
        d["loss_func"] = func.__name__
        df.append(d)
        for k in rep_dict.keys():
            d = data.copy()
            d["loss"] = func(rep_dict[k], met_rep_dict[k])
            d["loss_type"] = k
            d["loss_func"] = func.__name__
            df.append(d)

    df = pd.DataFrame(df)
    df.to_csv(output_dir / "loss.csv", index=False)
    torch.save(met.model(met.metamer), output_dir / "rep.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand checkerboard synth",
    )
    parser.add_argument("--img", "-i", default="checkerboard")
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--max_iter", default=20, type=int)
    parser.add_argument("--max_eval", default=None)
    parser.add_argument("--history_size", default=100, type=int)
    parser.add_argument("--line_search_fn", default=None,)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", default=200, type=int)
    parser.add_argument("--lr", default=.01, type=float)
    parser.add_argument("--output_dir", '-o', default=".")
    parser.add_argument("--mn", type=float, default=1.)
    parser.add_argument("--mag_std", type=float, default=1.)
    parser.add_argument("--autocorr_recon", type=float, default=1.)
    parser.add_argument("--skew_recon", type=float, default=1.)
    parser.add_argument("--kurt_recon", type=float, default=1.)
    parser.add_argument("--cross_ori", type=float, default=1.)
    parser.add_argument("--cross_scale_mag", type=float, default=1.)
    parser.add_argument("--cross_scale_real", type=float, default=1.)
    parser.add_argument("--autocorr_mag", type=float, default=1.)
    args = vars(parser.parse_args())
    try:
        args["max_eval"] = int(args["max_eval"])
    except:
        # then this is None
        args["max_eval"] = None
    if args["line_search_fn"] == "None":
        args["line_search_fn"] = None
    main(**args)
