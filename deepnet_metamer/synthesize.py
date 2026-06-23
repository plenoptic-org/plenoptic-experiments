"""From Feather et al. 2023."""

# %timeit for layer 2 model(img):
# cpu, float32: 14 msec
# gpu, float32: 4 msec
# cpu, float64: 26 msec
# gpu, float64: 22 msec
# %timeit for layer 3 model(img):
# cpu, float32: 26 msec
# gpu, float32: 6 msec
# cpu, float64: 43 msec
# gpu, float64: 57 msec
# %timeit for layer 4 model(img):
# cpu, float32: 36 msec
# gpu, float32: 8 msec
# cpu, float64: 50 msec
# gpu, float64: 90 msec

import argparse
import functools
import pathlib
import torch
import torchvision
import plenoptic as po
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(seed=None, image="parrot", layer="layer3", lr=0.01, max_iter=1000, save_path="metamer.pt", device="cpu",
         scheduler_name=None, optimizer_name="Adam", loss="mse", init_style="reduced"):

    if max_iter < 100:
        raise ValueError(f"max_iter must be at least 100, but got {max_iter=}")

    if seed is not None:
        po.set_seed(seed)
    save_path = pathlib.Path(save_path)

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    tv_model = torchvision.models.resnet50(weights=weights).eval()
    # This model's transform consists of resizing, cropping, and normalizing.
    # We recommend only including the normalizing in the transform.
    tv_transform = weights.transforms()
    norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
    model = po.models.FeatureExtractorModel(tv_model, layer, norm)
    po.remove_grad(model)
    model.to(device).to(torch.float64)

    if image == "parrot":
        img = po.data.parrot(False)
    elif image == "macaque":
        # get this down to approximately the right size, so it looks good when cropped
        img = po.load_images(po.data.fetch_data("Macaca_nigra_self-portrait.jpg"), False)
        img = po.process.blur_downsample(img, 2)[...,:-60,:]

    img = po.process.center_crop(img, tv_transform.crop_size[0])
    img = img.to(device).to(torch.float64)
    norm_rep = model(img).pow(2)
    if init_style == "reduced":
        init_img = 0.05 * torch.randn_like(img) + 0.5
    elif init_style == "full":
        init_img = torch.rand_like(img)

    def norm_mse(synth_rep, ref_rep, epsilon=1e-10):
        loss = (ref_rep - synth_rep).pow(2) / (norm_rep + epsilon)
        return loss.mean()

    def get_category(image):
        imagenet_categories = np.asarray(weights.meta["categories"])
        image_cat = po.to_numpy(torch.nn.functional.softmax(tv_model(norm(image)), dim=1).squeeze())
        return imagenet_categories[image_cat > 0.1]

    synth_kwargs = {"stop_iters_to_check": max_iter}
    if loss == "norm_mse":
        loss = norm_mse
    elif loss == "mse":
        loss = po.loss.mse
    elif loss == "l2_norm":
        loss = po.loss.l2_norm
    else:
        raise ValueError(f"Not sure how to handle {loss=}")

    met = po.Metamer(img, model, loss_function=loss)
    scheduler_kwargs = {}
    if scheduler_name is not None:
        if scheduler_name.startswith("StepLR"):
            scheduler_kwargs = {"step_size": int(scheduler_name.split("-")[1]), "gamma": 0.5}
            scheduler = torch.optim.lr_scheduler.StepLR
        else:
            raise ValueError(f"Not sure how to handle {scheduler_name=}")
    else:
        scheduler = None
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam
    elif optimizer_name == "LBFGS":
        optimizer = torch.optim.LBFGS
    else:
        raise ValueError(f"Not sure how to handle {optimizer=}")
    met.setup(init_img, optimizer_kwargs={"lr": lr}, scheduler_kwargs=scheduler_kwargs, scheduler=scheduler, optimizer=optimizer)
    met.synthesize(max_iter, store_progress=max_iter//100, **synth_kwargs)
    met.save(save_path)
    print(met.metamer.device)
    fig = po.plot.synthesis_status(met)

    orig_image_category = get_category(met.image)
    met_image_category = get_category(met.metamer)
    pearson_corr = torch.corrcoef(torch.cat([model(met.metamer), model(met.image)], 0))[0, 1].item()

    fig.text(0.5, 1.5, f"Image category: {orig_image_category}\nMetamer category:{met_image_category}\nPearson R: {pearson_corr}",
             ha="center", transform=fig.axes[0].transAxes)

    fig.savefig(save_path.with_suffix(".svg"))
    plt.close(fig)
    ax = po.plot.synthesis_imshow(met)
    ax.figure.savefig(save_path.with_suffix(".png"))
    po.plot.synthesis_animate(met).save(save_path.with_suffix(".mp4"))

    if len(met_image_category) == 0:
        met_image_category = "None"

    data = {"image_name": image, "model": "ResNet50", "layer": layer, "scheduler": scheduler_name, "optimizer": optimizer_name,
            "max_iter": max_iter, "lr": lr, "device": device, "loss_func": loss.__name__, "image_path": save_path.with_suffix(".svg"),
            "seed": seed, "loss": met.losses[-1].item(), "penalty": met.penalties[-1].item(),
            "orig_image_category": ",".join(orig_image_category), "met_image_category": ",".join(met_image_category),
            "pearson_corr": pearson_corr, "init_style": init_style}
    print(data)
    pd.DataFrame(data, index=[0]).to_csv(save_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some ResNet50 metamer generation",
    )
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--scheduler", "-c", default=None)
    parser.add_argument("--optimizer", "-o", default="Adam")
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--layer", default="layer3")
    parser.add_argument("--image", default="parrot")
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--lr", "-l", default=0.01, type=float)
    parser.add_argument("--max_iter", "-n", default=2000, type=int)
    parser.add_argument("--save_path", '-f', default="metamer.pt")
    parser.add_argument("--init_style", default="reduced")
    args = vars(parser.parse_args())
    print(args)
    device = args.pop("device")
    try:
        device = torch.device(device)
    except RuntimeError:
        device = torch.device(int(device))
    scheduler = args.pop("scheduler")
    if scheduler == "None":
        scheduler = None
    optimizer = args.pop("optimizer")
    main(device=device, scheduler_name=scheduler, optimizer_name=optimizer, **args)
