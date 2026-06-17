"""From Feather et al. 2023."""

import argparse
import functools
import pathlib
import re
import torch
import torchvision
import plenoptic as po
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(layer="layer3", load_path="metamer.pt", loss="mse", animate=False, plot=True, create_csv=False):

    if (animate + plot + create_csv) == 0:
        raise ValueError("Must do something!")

    device = "cpu"
    load_path = pathlib.Path(load_path)

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    tv_model = torchvision.models.resnet50(weights=weights).eval()
    # This model's transform consists of resizing, cropping, and normalizing.
    # We recommend only including the normalizing in the transform.
    tv_transform = weights.transforms()
    norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
    model = po.models.FeatureExtractorModel(tv_model, layer, norm)
    po.remove_grad(model)
    model.to(device).to(torch.float64)
    img = po.process.center_crop(po.data.parrot(False), tv_transform.crop_size[0])
    img = img.to(device).to(torch.float64)

    def norm_mse(synth_rep, ref_rep, epsilon=1e-10):
        loss = (ref_rep - synth_rep).pow(2) / (norm_rep + epsilon)
        return loss.mean()

    def get_category(image):
        imagenet_categories = np.asarray(weights.meta["categories"])
        image_cat = po.to_numpy(torch.nn.functional.softmax(tv_model(norm(image)), dim=1).squeeze())
        return imagenet_categories[image_cat > 0.1]

    if loss == "norm_mse":
        loss = norm_mse
    elif loss == "mse":
        loss = po.loss.mse
    elif loss == "l2_norm":
        loss = po.loss.l2_norm
    else:
        raise ValueError(f"Not sure how to handle {loss=}")

    met = po.Metamer(img, model, loss_function=loss)
    met.load(load_path, map_location=device)

    orig_image_category = get_category(met.image)
    met_image_category = get_category(met.metamer)
    pearson_corr = torch.corrcoef(torch.cat([model(met.metamer), model(met.image)], 0))[0, 1].item()

    if plot:
        fig = po.plot.synthesis_status(met)

        fig.text(0.5, 1.5, f"Image category: {orig_image_category}\nMetamer category:{met_image_category}\nPearson R: {pearson_corr}",
                 ha="center", transform=fig.axes[0].transAxes)

        fig.savefig(load_path.with_suffix(".svg"))
        plt.close(fig)
        ax = po.plot.synthesis_imshow(met)
        ax.figure.savefig(load_path.with_suffix(".png"), bbox_inches="tight")
    if animate:
        po.plot.synthesis_animate(met).save(load_path.with_suffix(".mp4"))
    if create_csv:
        try:
            scheduler_name = f'{met.scheduler[0].split(".")[-1]}-{met.scheduler[1]["step_size"]}'
        except AttributeError:
            scheduler_name = "None"
        optimizer_name = met.optimizer[0].split(".")[-1]
        lr = float(re.findall(r"lr-([0-9e.-]+)_", str(load_path))[0])
        seed = int(re.findall(r"seed-([0-9]+)_", str(load_path))[0])
        data = {"image_name": "parrot", "model": "ResNet50", "layer": layer, "scheduler": scheduler_name, "optimizer": optimizer_name,
                "max_iter": len(met.losses)-1, "lr": lr, "device": device, "loss_func": loss.__name__, "image_path": load_path.with_suffix(".svg"),
                "seed": seed, "loss": met.losses[-1].item(), "penalty": met.penalties[-1].item(),
                "orig_image_category": orig_image_category, "met_image_category": met_image_category, "pearson_corr": pearson_corr}
        pd.DataFrame(data, index=[0]).to_csv(load_path.with_suffix(".csv"), index=False)
        print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some ResNet50 metamer generation",
    )
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--layer", default="layer3")
    parser.add_argument("--load_path", '-f', default="metamer.pt")
    parser.add_argument("--create_csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--animate", action="store_true")
    args = vars(parser.parse_args())
    print(args)
    main(**args)
