"""Synthesize color PS metamers from uniform and color-matched noise."""

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
metamer_dir = Path("metamers")
metamer_dir.mkdir(exist_ok=True)

for image_name in image_names:
    # Load the target image in the 0--1 range expected by plenoptic
    image_path = validation_dir / "inputs" / f"{image_name}.tif"
    image = po.load_images(image_path, as_gray=False).to(device=device, dtype=dtype)

    # Set up the model with the PCA transform of the target, and the default loss
    model = po.models.PortillaSimoncelli(
        image.shape[-2:],
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=7,
        color_statistics=True,
        transform=po.process.PCA(image, max_relative_scaling=float("inf")),
    ).to(device=device, dtype=dtype)
    model.eval()
    po.remove_grad(model)
    loss = po.loss.portilla_simoncelli_loss_factory(model, image)

    # Synthesize from uniform noise (Metamer's default) and color-matched noise
    initial_images = {
        "uniform": None,
        "color_matched": color_matched_noise(image),
    }
    for condition, initial_image in initial_images.items():
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
