import plenoptic as po
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import torch
import argparse
import itertools
import pathlib
from typing import Literal
import imageio.v3 as iio
import pandas as pd
import seaborn as sns
import pyiqa

N_IMGS = 2

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72

def init_dists(img):
    metric = pyiqa.create_metric("dists", device=img.device, as_loss=True)
    po.remove_grad(metric)
    metric.to(img.dtype)

    def dists(x):
        return metric(x[:1], x[-1:])

    return dists


def pairwise_image_mse(imgs):
    """Get pair-wise MSE between images, along batch dimension."""
    n = imgs.shape[0]
    idx = itertools.combinations(range(n), 2)
    mse = torch.nan * torch.zeros((n, n))
    for i, j in idx:
        mse[j, i] = po.loss.mse(imgs[i], imgs[j])
    return po.to_numpy(mse)


def create_metamer_figure(met):
    n_cols = len(met.image) + 2
    zoom = 256 // met.image.shape[-1]
    fig = plt.figure(figsize=(5 * n_cols + 2, 16))
    gs = mpl.gridspec.GridSpec(3, n_cols, figure=fig)
    im_axes = [fig.add_subplot(gs[0, i]) for i in range(N_IMGS + 1)]
    im_axes += [fig.add_subplot(gs[1, i]) for i in range(N_IMGS + 1)]
    # met.image has more than one dimension, but they're all identical, so just use the
    # first. image[:1] is the same as image[0], but preserves the number of dimensions.
    imgs = torch.cat([met.image[:1], met.metamer])
    # concatenate the representation of those images
    reps = met.model(imgs)
    titles = ["Target image"] + [f"Model metamer[{i}]" for i in range(N_IMGS)]
    titles += ["Representation of target"] + [
        f"Representation of metamer[{i}]" for i in range(N_IMGS)
    ]
    for ax, im, t in zip(im_axes, torch.cat([imgs, reps]), titles):
        vr = (0, 1) if "Representation" not in t else "indep1"
        po.plot.imshow(im.unsqueeze(0), ax=ax, title=t, zoom=zoom, vrange=vr)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    imgs_mse = pairwise_image_mse(imgs)
    reps_mse = pairwise_image_mse(reps)
    labels = ["Target"] + [f"Metamer[{i}]" for i in range(N_IMGS)]
    ax = fig.add_subplot(gs[0, -1])
    sns.heatmap(imgs_mse, ax=ax, annot=True, xticklabels=labels, yticklabels=labels)
    ax.set_title("MSE of Images")
    ax = fig.add_subplot(gs[1, -1])
    sns.heatmap(reps_mse, ax=ax, annot=True, xticklabels=labels, yticklabels=labels)
    ax.set_title("MSE of Representations")
    po.plot.synthesis_loss(met, ax=fig.add_subplot(gs[2, 0:2]), plot_penalties=True)
    return fig

def init_metamer(
    img: str,
    model: str,
    penalty: str = "dists",
    penalty_lambda: float = 0.1,
    loss_func: str = "mse",
    lr: float = 0.01,
    init_seed: int = 0,
    device: str | int = 0,
):
    try:
        device = int(device)
    except:
        pass
    device = torch.device(device)
    if "blur" in img:
        img, mod = img.split("-")
        mod = mod.replace("blur", "")
        img = eval(f"po.data.{img}()")
        img = po.process.blur_downsample(img, int(mod))
    elif "crop" in img:
        img, mod = img.split("-")
        mod = mod.replace("crop", "")
        img = eval(f"po.data.{img}()")
        img = po.process.center_crop(img, int(mod))
    else:
        img = eval(f"po.data.{img}()")
    img = img.to(device).to(torch.float64)
    po.set_seed(init_seed)
    if model == "LGC":
        model = po.models.LuminanceGainControl(
            kernel_size=(31, 31), pad_mode="circular",
            pretrained=True, cache_filt=True
        )
        loss = eval(f"po.loss.{loss_func}")
        opt_kwargs = {"lr": lr}
        optim = torch.optim.Adam
        model.to(device).to(torch.float64)

    if penalty == "dists":
        dists = init_dists(img)
        def penalty(x):
            return po.regularize.penalize_range(x) + torch.exp(-dists(x))
    elif penalty == "range":
        penalty = po.regularize.penalize_range
    else:
        raise ValueError(f"{penalty=}")

    model.eval()
    po.remove_grad(model)
    met = po.Metamer(img.repeat(2, 1, 1, 1), model, loss_function=loss,
                     penalty_function=penalty, penalty_lambda=penalty_lambda)
    print(met)
    return met, optim, opt_kwargs


def main(
    img: str,
    model: str,
    penalty: str = "dists",
    penalty_lambda: float = 0.1,
    init_seed: int = 0,
    lr: float = 0.01,
    loss_func: str = "l2_norm",
    device: str | int = 0,
    synth_max_iter: int = 200,
    two_stage: bool = False,
    output_path: str | pathlib.Path = "result.pt",
):
    torch.set_num_threads(1)
    output_path = pathlib.Path(output_path)
    data = {"target_image": img, "penalty_lambda": penalty_lambda, "lr": lr, "loss_func": loss_func}
    if not two_stage:
        met, optim, opt_kwargs = init_metamer(img, model, penalty, penalty_lambda, loss_func, lr,
                                              init_seed, device)
        met.setup(optimizer=optim, optimizer_kwargs=opt_kwargs)
        met.synthesize(max_iter=synth_max_iter, stop_criterion=1e-16, stop_iters_to_check=1000,
                       )
    else:
        try:
            loss_func1, loss_func2 = loss_func.split("-")
        except ValueError:
            loss_func1 = loss_func2 = loss_func
        try:
            lr1, lr2 = lr.split("-")
            lr1 = float(lr1)
            lr2 = float(lr2)
        except (ValueError, AttributeError):
            lr1 = lr2 = float(lr)
        met, optim, opt_kwargs = init_metamer(img, model, "range", 0.1, loss_func1, lr1,
                                              init_seed, device)
        met.setup(optimizer=optim, optimizer_kwargs=opt_kwargs)
        met.synthesize(max_iter=synth_max_iter//2, stop_criterion=1e-16, stop_iters_to_check=1000,
                       )
        init_img = met.metamer
        met, optim, opt_kwargs = init_metamer(img, model, penalty, penalty_lambda, loss_func2, lr2,
                                              init_seed, device)
        met.setup(optimizer=optim, optimizer_kwargs=opt_kwargs, initial_image=init_img)
        met.synthesize(max_iter=synth_max_iter//2, stop_criterion=1e-16, stop_iters_to_check=1000,
                       )
    print(f"saving to {output_path}")
    met.save(output_path)
    dists = init_dists(met.image)
    dist_val = dists(met.metamer).item()
    data["dists"] = dist_val
    data["obj_func"] = met.losses[-1].item()
    data["metamer_loss"] = po.loss.mse(met.model(met.image), met.model(met.metamer)).item()
    data["image_path"] = output_path.with_suffix(".png")
    pd.DataFrame(data, index=[0]).to_csv(output_path.with_suffix(".csv"), index=False)
    met.to("cpu")
    fig = create_metamer_figure(met)
    fig.suptitle(f"DISTS={dist_val}", y=0.95, fontsize="xx-large")
    fig.savefig(output_path.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand LBFGS",
    )
    parser.add_argument("--img", "-i", default="einstein")
    parser.add_argument("--model", "-m", default="LGC")
    parser.add_argument("--penalty", "-p", default="dists")
    parser.add_argument("--penalty_lambda", "-l", type=float, default=0.1)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--loss_func", default="l2_norm")
    parser.add_argument("--init_seed", "-s", type=int, default=0)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", "-n", default=200, type=int)
    parser.add_argument("--output_path", '-f', default="result.pt")
    parser.add_argument("--two-stage", action="store_true")
    args = vars(parser.parse_args())
    device = args.pop("device")
    try:
        device = torch.device(device)
    except RuntimeError:
        device = torch.device(int(device))
    main(device=device, **args)
