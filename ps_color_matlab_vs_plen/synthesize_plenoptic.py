from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

import plenoptic as po

# Texture to synthesize and directory to save the plenoptic metamers
image_name = "DSCF4315"
image_path = Path(f"inputs/{image_name}.tif")
metamer_dir = Path("plenoptic_metamers")
metamer_dir.mkdir(exist_ok=True)

# Synthesis settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
seed = 0
n_synthesis_iterations = 100


def color_matched_noise(target):
    """Gaussian noise with exactly the target RGB population covariance and mean."""
    po.set_seed(seed)
    noise = torch.randn_like(target)
    noise_pixels = noise.permute(0, 2, 3, 1).reshape(-1, 3)
    noise_pixels = noise_pixels - noise_pixels.mean(0)
    noise_covariance = noise_pixels.mT @ noise_pixels / noise_pixels.shape[0]
    noise_eigenvalues, noise_eigenvectors = torch.linalg.eigh(noise_covariance)
    whitening = (
        noise_eigenvectors
        @ torch.diag(noise_eigenvalues.rsqrt())
        @ noise_eigenvectors.mT
    )

    target_pixels = target.permute(0, 2, 3, 1).reshape(-1, 3)
    target_mean = target_pixels.mean(0)
    target_pixels = target_pixels - target_mean
    target_covariance = target_pixels.mT @ target_pixels / target_pixels.shape[0]
    target_eigenvalues, target_eigenvectors = torch.linalg.eigh(target_covariance)
    coloring = (
        target_eigenvectors
        @ torch.diag(target_eigenvalues.sqrt())
        @ target_eigenvectors.mT
    )

    noise_pixels = noise_pixels @ whitening @ coloring + target_mean
    return (
        noise_pixels.reshape(target.shape[0], *target.shape[-2:], 3)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


# Load the image in the 0--1 range expected by plenoptic
image = po.load_images(image_path, as_gray=False).to(device=device, dtype=dtype)

# Set up both color transforms and the two initialization conditions
transforms = {
    "pca": po.process.PCA(image, max_relative_scaling=float("inf")),
    "opc": po.process.OPC(),
}
initial_images = {
    "default": None,
    "color_matched": color_matched_noise(image),
}

# Configure the LBFGS optimizer used for synthesis
optimizer_kwargs = {
    "max_iter": 10,
    "max_eval": 10,
    "history_size": 100,
    "line_search_fn": "strong_wolfe",
    "lr": 1,
}

# Synthesize and save one metamer for each transform and initialization
for color_space, transform in transforms.items():
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

    loss = po.loss.portilla_simoncelli_loss_factory(model, image)
    for initialization, initial_image in initial_images.items():
        po.set_seed(seed)
        metamer = po.Metamer(image, model, loss_function=loss)
        metamer.setup(
            initial_image=initial_image,
            optimizer=torch.optim.LBFGS,
            optimizer_kwargs=optimizer_kwargs,
        )
        metamer.synthesize(max_iter=n_synthesis_iterations)

        metamer_array = po.to_numpy(metamer.metamer.squeeze(0).permute(1, 2, 0))
        metamer_uint8 = po.convert_float_to_int(np.clip(metamer_array, 0, 1))
        metamer_path = (
            metamer_dir
            / f"plenoptic_{color_space}_{initialization}_{image_name}.tif"
        )
        iio.imwrite(metamer_path, metamer_uint8)
        print(
            f"Saved {color_space.upper()} {initialization} plenoptic metamer "
            f"to {metamer_path}"
        )

        del metamer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, loss

print(f"Ran all syntheses on {device} with {dtype}")
