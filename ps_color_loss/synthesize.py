"""Synthesize color PS metamers from different initializations and losses."""

import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

import plenoptic as po

# Reuse the settings and initialization of the validation experiment
validation_dir = Path("../ps_color_matlab_vs_plen")
sys.path.append(str(validation_dir))
from synthesize_plenoptic import (  # noqa: E402
    color_matched_noise,
    device,
    dtype,
    optimizer_kwargs,
    seed,
)

image_names = json.loads((validation_dir / "sweep.json").read_text())["images"]
n_iterations = 100
mean_weights = [10, 100, 250, 1000]
# Color transform used by the model during synthesis, "pca" or "opc"
color_transform = "pca"
metamer_dir = Path("metamers") / color_transform
metamer_dir.mkdir(parents=True, exist_ok=True)

for image_name in image_names:
    # Load the target image in the 0--1 range expected by plenoptic
    image_path = validation_dir / "inputs" / f"{image_name}.tif"
    image = po.load_images(image_path, as_gray=False).to(device=device, dtype=dtype)

    # Set up the model with the chosen color transform
    if color_transform == "pca":
        transform = po.process.PCA(image, max_relative_scaling=float("inf"))
    else:
        transform = po.process.OPC()
    model = po.models.PortillaSimoncelli(
        image.shape[-2:],
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=7,
        color_statistics=True,
        transform=transform,
    ).to(device=device, dtype=dtype)
    model.eval()
    po.remove_grad(model)

    # Each condition has an initial image and a loss reweighting. Uniform noise
    # (initial image None) is Metamer's default, and reweighting None the default loss
    conditions = {
        "uniform": (None, None),
        "color_matched": (color_matched_noise(image), None),
    }
    # Weight the RGB means more. Like the default loss, leave min/max out and
    # weight the highpass variance 100 times.
    # Pixel statistics are mean, variance, skew, kurtosis, min and max
    for mean_weight in mean_weights:
        pixel_weights = torch.tensor(
            [mean_weight, 1, 1, 1, 0, 0], dtype=dtype, device=device
        )
        conditions[f"mean_weight_{mean_weight}"] = (
            None,
            {"pixel_statistics": pixel_weights, "var_highpass_residual": 100},
        )

    for condition, (initial_image, reweighting_dict) in conditions.items():
        loss = po.loss.portilla_simoncelli_loss_factory(model, image, reweighting_dict)
        po.set_seed(seed)
        metamer = po.Metamer(image, model, loss_function=loss)
        metamer.setup(
            initial_image=initial_image,
            optimizer=torch.optim.LBFGS,
            optimizer_kwargs=optimizer_kwargs,
        )
        metamer.synthesize(max_iter=n_iterations)

        # Save the metamer as an 8-bit image
        metamer_array = po.to_numpy(metamer.metamer.squeeze(0).permute(1, 2, 0))
        metamer_uint8 = po.convert_float_to_int(np.clip(metamer_array, 0, 1))
        metamer_path = metamer_dir / f"{condition}_{image_name}.tif"
        iio.imwrite(metamer_path, metamer_uint8)
        print(f"Saved {condition} metamer to {metamer_path}")
