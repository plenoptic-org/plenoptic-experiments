import plenoptic as po
import matplotlib.pyplot as plt
from plenoptic.data.fetch import fetch_data
import torch
import argparse
import itertools
import pathlib
from typing import Literal
import imageio.v3 as iio
import pandas as pd
from plot import compute, plot, animate


def spyr_loss(img, device, variant):
    pyr = po.simul.SteerablePyramidFreq(img.shape[-2:])
    pyr.to(device).to(torch.float64)
    coeffs = {k: v.abs().mean(dim=(-2, -1)) for k, v in pyr(img).items()}
    coeffs = torch.cat([v.flatten(1, -1) for v in coeffs.values()], 1)
    if variant.startswith("exp"):
        func = torch.exp
    else:
        func = lambda x: x
    variant = variant.replace("exp", "")
    if "mask" in variant:
        mask = pyr(img)
        if variant == "maskall":
            for scale in [1, 2, 3]:
                if scale in mask:
                    mask[scale] = torch.zeros_like(mask[scale])
        elif variant == "maskhigh":
            for scale in [1, 2, 3, 4, 5, "residual_lowpass"]:
                if scale in mask:
                    mask[scale] = torch.zeros_like(mask[scale])
        elif variant == "masklow":
            for scale in ["residual_highpass", 0, 1, 2, 3]:
                if scale in mask:
                    mask[scale] = torch.zeros_like(mask[scale])
        elif variant == "maskvert":
            for scale in ["residual_highpass", "residual_lowpass", 1, 2, 3]:
                if scale in mask:
                    mask[scale] = torch.zeros_like(mask[scale])
            for scale in [0, 4, 5]:
                if scale in mask:
                    mask[scale][:, :, 1:] = 0
        elif variant == "maskdiag":
            for scale in ["residual_highpass", "residual_lowpass", 1, 2, 3]:
                if scale in mask:
                    mask[scale] = torch.zeros_like(mask[scale])
            for scale in [0, 4, 5]:
                if scale in mask:
                    mask[scale][:, :, 0] = 0
                    mask[scale][:, :, 2] = 0
        mask = {k: v.abs().mean(dim=(-2, -1)) for k, v in mask.items()}
        mask = torch.cat([v.flatten(1, -1) for v in mask.values()], 1)
        mask = mask.to(torch.bool)
    else:
        mask = torch.ones_like(coeffs)

    def penalty(x):
        penalty = {k: v.abs().mean(dim=(-2, -1)) for k, v in pyr(x).items()}
        penalty = torch.cat([v.flatten(1, -1) for v in penalty.values()], 1)
        penalty = mask * (penalty / coeffs)
        return func(1 - penalty.diff(dim=0).pow(2).mean())

    return penalty


def alexnet_category_loss(comb_func, device):
    import torchvision
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    # THIS IS IMPORTANT in order to prevent dropout and thus make model output non-stochastic
    model.eval()
    transform = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
    # for some reason, transfrom will always make it a float32 again, so no reason to
    # convert model to float64...
    model.to(device)

    def penalty(x):
        # while softmax of last layer is present in every diagram of AlexNet I can find,
        # torchvision's implementation doesn't seem to do so
        model_out = torch.nn.functional.softmax(model(transform(x.repeat(1, 3, 1, 1))), dim=1)
        norms = torch.linalg.vector_norm(model_out, 2, dim=-1)
        penalty = [torch.dot(model_out[i], model_out[j]) / (norms[i] * norms[j])
                   for i, j in itertools.combinations(range(x.shape[0]), 2)]
        return comb_func(1 - torch.stack(penalty))

    return penalty


def init_metamer(
    img: str,
    model: str,
    penalty: str = "pixel",
    penalty_lambda: float = 0.1,
    batch_n: int = 2,
    init_seed: int = 0,
    device: str | int = 0,
):
    if batch_n < 2:
        raise ValueError(f"batch_n must be >= 2 but got {batch_n}!")
    try:
        device = int(device)
    except:
        pass
    device = torch.device(device)
    if "blur" in img:
        img, mod = img.split("-")
        mod = mod.replace("blur", "")
        img = eval(f"po.data.{img}()")
        img = po.tools.blur_downsample(img, int(mod))
    elif "crop" in img:
        img, mod = img.split("-")
        mod = mod.replace("crop", "")
        img = eval(f"po.data.{img}()")
        img = po.tools.center_crop(img, int(mod))
    img = img.to(device).to(torch.float64)
    po.tools.set_seed(init_seed)
    if model == "LGC":
        model = po.simul.LuminanceGainControl(
            kernel_size=(31, 31), pad_mode="circular",
            pretrained=True, cache_filt=True
        )
        loss = po.tools.optim.mse
        opt_kwargs = {}
        optim = torch.optim.Adam
        model.to(device).to(torch.float64)
    elif model == "PS":
        model = po.simul.PortillaSimoncelli(img.shape[2:])
        model.to(device).to(torch.float64)
        loss = po.tools.optim.portilla_simoncelli_loss_factory(model, img)
        opt_kwargs = {
            "max_iter": 10,
            "max_eval": 10,
            "history_size": 100,
            "line_search_fn": "strong_wolfe",
            "lr": 1,
        }
        optim = torch.optim.LBFGS
    try:
        penalty, variant = penalty.split("-")
        if variant == "logsumexp":
            comb_func = lambda x: torch.logsumexp(x, 0)
        elif variant == "sse":
            comb_func = lambda x: x.pow(2).sum()
        if penalty == "mse":
            def penalty_part(x):
                penalty = [po.tools.optim.mse(x.select(0, i), x.select(0, j)) for i, j in
                           itertools.combinations(range(x.shape[0]), 2)]
                return comb_func(1-torch.stack(penalty))
        elif penalty == "alexnet":
            penalty_part = alexnet_category_loss(comb_func, device)
        elif penalty == "spyr":
            penalty_part = spyr_loss(img, device, variant)
        else:
            func = eval(f"po.metric.{penalty}")
            if penalty == "nlpd":
                transform = lambda x: 2-x
            else:
                transform = lambda x: x
            def penalty_part(x):
                penalty = [func(x.select(0, i).unsqueeze(0), x.select(0, j).unsqueeze(0))
                           for i, j in itertools.combinations(range(x.shape[0]), 2)]
                return comb_func(transform(torch.stack(penalty)))
        def penalty(x):
            return po.tools.regularization.penalize_range(x) + penalty_part(x)
    except ValueError:
        penalty = po.tools.regularization.penalize_range
    model.eval()
    po.tools.remove_grad(model)
    met = po.synth.Metamer(img.repeat(batch_n, 1, 1, 1), model, loss_function=loss,
                           penalty_function=penalty, penalty_lambda=penalty_lambda)
    return met, optim, opt_kwargs


def main(
    img: str,
    model: str,
    penalty: str = "pixel",
    penalty_lambda: float = 0.1,
    batch_n: int = 2,
    init_seed: int = 0,
    device: str | int = 0,
    synth_max_iter: int = 200,
    output_path: str | pathlib.Path = "result.pt",
):
    torch.set_num_threads(1)
    output_path = pathlib.Path(output_path)
    met, optim, opt_kwargs = init_metamer(img, model, penalty, penalty_lambda,
                                          batch_n, init_seed, device)
    met.setup(optimizer=optim, optimizer_kwargs=opt_kwargs)
    met.synthesize(max_iter=synth_max_iter, stop_criterion=1e-16, stop_iters_to_check=1000,
                   store_progress=synth_max_iter//50)
    print(f"saving to {output_path}")
    met.save(output_path)
    met_loss, met_penalty = compute(met, device)
    met.to("cpu")
    plot(met, met_loss, met_penalty, output_path.with_suffix(".svg"))
    # animate(met, output_path.with_name(f"{output_path.stem}-0.mp4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand LBFGS",
    )
    parser.add_argument("--img", "-i", default="einstein")
    parser.add_argument("--model", "-m", default="LGC")
    parser.add_argument("--penalty", "-p", default="pixel")
    parser.add_argument("--penalty_lambda", "-l", type=float, default=.1)
    parser.add_argument("--batch_n", "-b", type=int, default=2)
    parser.add_argument("--init_seed", "-s", type=int, default=0)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", "-n", default=200, type=int)
    parser.add_argument("--output_path", '-f', default="result.pt")
    args = vars(parser.parse_args())
    device = args.pop("device")
    try:
        device = torch.device(device)
    except RuntimeError:
        device = torch.device(int(device))
    main(device=device, **args)
