"""Compare the RGB mean and covariance of the metamers with those of their targets."""

import json
import sys
from pathlib import Path

import torch

import plenoptic as po

# Reuse the CSV writer of the validation experiment
validation_dir = Path("../ps_color_matlab_vs_plen")
sys.path.append(str(validation_dir))
from compare_statistics import write_results  # noqa: E402

image_names = json.loads((validation_dir / "sweep.json").read_text())["images"]
conditions = [
    "uniform",
    "color_matched",
    "mean_weight_10",
    "mean_weight_100",
    "mean_weight_250",
    "mean_weight_1000",
]


def rgb_mean_and_covariance(path):
    """Load an RGB image and return its channel means and 3 x 3 covariance."""
    image = po.load_images(path, as_gray=False).double()
    pixels = image.flatten(2).squeeze(0)
    mean = pixels.mean(dim=-1)
    centered = pixels - mean[:, None]
    covariance = centered @ centered.T / pixels.shape[-1]
    return mean, covariance


rows = []
for image_name in image_names:
    # Collect the target, the MATLAB metamer and the plenoptic metamers
    paths = {
        "target": validation_dir / "inputs" / f"{image_name}.tif",
        "matlab": validation_dir / "matlab_metamers" / f"matlab_{image_name}.tif",
    }
    for condition in conditions:
        paths[condition] = Path("metamers") / f"{condition}_{image_name}.tif"

    target_mean, target_covariance = rgb_mean_and_covariance(paths["target"])
    for condition, path in paths.items():
        mean, covariance = rgb_mean_and_covariance(path)
        # Largest channel mean difference, and covariance error relative to target
        mean_error = (mean - target_mean).abs().max()
        covariance_error = torch.linalg.norm(
            covariance - target_covariance
        ) / torch.linalg.norm(target_covariance)
        rows.append(
            {
                "image": image_name,
                "condition": condition,
                "mean_r": mean[0].item(),
                "mean_g": mean[1].item(),
                "mean_b": mean[2].item(),
                "mean_error": mean_error.item(),
                "covariance_error": covariance_error.item(),
            }
        )

write_results(rows, Path("results") / "color_metrics.csv")

# Print a summary table
print(
    f"{'image':<10} {'condition':<14} {'mean RGB':<20} {'mean err':>8} {'cov err':>8}"
)
for row in rows:
    mean_rgb = f"{row['mean_r']:.3f} {row['mean_g']:.3f} {row['mean_b']:.3f}"
    print(
        f"{row['image']:<10} {row['condition']:<14} {mean_rgb:<20} "
        f"{row['mean_error']:>8.4f} {row['covariance_error']:>8.4f}"
    )
