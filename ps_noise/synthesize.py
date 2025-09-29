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


def plot(model, met_img, init_img, target_img, synth_time, weights, device, seed,
         noise, save_path):
    weights = ",".join([str(int(i.item())) for i in weights])

    fig, axes = plt.subplots(1, 4, figsize=(32, 5), gridspec_kw={"width_ratios": [1, 1, 1, 3.1]})
    loss = po.tools.optim.l2_norm(model(target_img), model(met_img))
    titles = ["Original image", f"Initial noise {noise}\nseed-{seed}", f"Metamer\n{loss}"]
    for im, ax, t in zip([target_img, init_img, met_img], axes, titles):
        po.imshow(im, ax=ax, title=t)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    model.plot_representation(model(target_img) - model(met_img), ax=axes[-1], ylim=False)
    axes[-1].set_title(f"PS(target) - PS(metamer)\nWeight: {weights}", y=1.05)
    fig.suptitle(f"Device: {device}, duration: {synth_time}", fontsize='xx-large', y=1.1)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main(
    init_noise: float,
    img: str,
    init_noise_type: Literal["uniform", "normal"] = "uniform",
    init_seed: int = 0,
    weighted: list[int] = [1, 1, 1, 1, 1, 1, 10],
    history_size: int = 100,
    device: str | int = 0,
    synth_max_iter: int = 200,
    output_path: str = "result.pt",
):
    print(po)
    torch.set_num_threads(1)
    start = time.time()
    try:
        device = int(device)
    except:
        pass
    device = torch.device(device)
    if img == "einstein":
        img = po.data.einstein()
    elif img == "curie":
        img = po.data.curie()
    img = img.to(device).to(torch.float64)
    model = po.simul.PortillaSimoncelli(img.shape[-2:])
    model.to(device)
    po.tools.set_seed(init_seed)
    if init_noise_type == "normal":
        init_img = init_noise * torch.randn_like(img) + img
    elif init_noise_type == "uniform":
        init_img = init_noise * torch.rand_like(img)
    loss = po.tools.optim.l2_norm
    if weighted:
        weighted = torch.as_tensor(weighted).to(device)
        weights = model.convert_to_dict(torch.ones_like(model(img)))
        weights["pixel_statistics"] = weighted[:6] * torch.ones_like(weights["pixel_statistics"])
        weights["var_highpass_residual"] = weighted[-1] * torch.ones_like(weights["var_highpass_residual"])
        weights = model.convert_to_tensor(weights)
        def loss(x, y):
            return po.tools.optim.l2_norm(weights*x, weights*y)
    met = po.synth.Metamer(
        img,
        model,
        loss_function=loss,
        allowed_range=(min(0, init_img.min()), max(init_img.max(), 1)),
        range_penalty_lambda=0,
    )
    opt_kwargs = {"max_iter": 10, "max_eval": 10,
                  "history_size": history_size, "line_search_fn": "strong_wolfe",
                  "lr": 1}
    met.setup(init_img, optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
    init_stop = time.time()
    met.synthesize(max_iter=synth_max_iter, stop_criterion=1e-16, stop_iters_to_check=1000)
    synth_stop = time.time()
    save_dict = {"initial_time": init_stop - start,
                 "synth_time": synth_stop - start,
                 "metamer": met.metamer}
    output_path = pathlib.Path(output_path)
    plot(model, met.metamer, init_img, img, save_dict["synth_time"] - save_dict["initial_time"],
         weighted, device.type, init_seed, f"{init_noise_type}-{init_noise}", output_path.with_suffix(".svg"))
    torch.save(save_dict, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand LBFGS",
    )
    parser.add_argument("--img", "-i", default="fig4a")
    parser.add_argument("--init_seed", "-s", type=int)
    parser.add_argument("--init_noise", "-n", type=float)
    parser.add_argument("--init_noise_type", "-t", default="uniform")
    parser.add_argument("--weighted", default=None)
    parser.add_argument("--history_size", default=100, type=int)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", default=200, type=int)
    parser.add_argument("--output_path", '-f', default="result.pt")
    args = vars(parser.parse_args())
    weighted = args.pop("weighted")
    if weighted is None:
        weighted = [1, 1, 1, 1, 1, 1, 10]
    else:
        weighted = [float(i) for i in weighted.split(',')]
        assert len(weighted) == 7
    main(weighted=weighted, **args)
