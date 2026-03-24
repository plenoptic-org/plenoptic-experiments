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
import torchvision


def alexnet_category_loss(comb_func, device):
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
    img = eval(f"po.data.{img}()")
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
        penalty, comb_func = penalty.split("-")
        if comb_func == "logsumexp":
            comb_func = lambda x: torch.logsumexp(x, 0)
        elif comb_func == "sse":
            comb_func = lambda x: x.pow(2).sum()
        if penalty == "mse":
            def penalty_part(x):
                penalty = [po.tools.optim.mse(x.select(0, i), x.select(0, j)) for i, j in
                           itertools.combinations(range(x.shape[0]), 2)]
                return comb_func(1-torch.stack(penalty))
        elif penalty == "alexnet":
            penalty_part = alexnet_category_loss(comb_func, device)
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
    output_path: str = "result.pt",
):
    torch.set_num_threads(1)
    met, optim, opt_kwargs = init_metamer(img, model, penalty, penalty_lambda,
                                          batch_n, init_seed, device)
    met.setup(optimizer=optim, optimizer_kwargs=opt_kwargs)
    met.synthesize(max_iter=synth_max_iter, stop_criterion=1e-16, stop_iters_to_check=1000,
                   store_progress=synth_max_iter//50)
    print(f"saving to {output_path}")
    met.save(output_path)


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
    main(**args)
