"""Compare the RGB mean, RGB covariance and PS statistics of metamers and targets."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import plenoptic as po

# Reuse the metamer comparison and CSV writer of the validation experiment
validation_dir = Path("../ps_color_matlab_vs_plen")
sys.path.append(str(validation_dir))
from compare_metamers import family_errors, plot_errors  # noqa: E402
from compare_statistics import statistics_to_compare, write_results  # noqa: E402

image_names = json.loads((validation_dir / "sweep.json").read_text())["images"]
conditions = [
    "uniform",
    "color_matched",
    "mean_weight_10",
    "mean_weight_100",
    "mean_weight_250",
    "mean_weight_1000",
]
# Color transform used during synthesis, "pca" or "opc". Evaluation always uses PCA
color_transform = "pca"
metamer_dir = Path("metamers") / color_transform
results_dir = Path("results") / color_transform


def rgb_mean_and_covariance(image):
    """Return the channel means and 3 x 3 covariance of an RGB image."""
    pixels = image.flatten(2).squeeze(0)
    mean = pixels.mean(dim=-1)
    centered = pixels - mean[:, None]
    covariance = centered @ centered.T / pixels.shape[-1]
    return mean, covariance


color_rows = []
family_rows = []
all_images = {}
for image_name in image_names:
    # Load the target, the MATLAB metamer and the plenoptic metamers
    paths = {
        "target": validation_dir / "inputs" / f"{image_name}.tif",
        "matlab": validation_dir / "matlab_metamers" / f"matlab_{image_name}.tif",
    }
    for condition in conditions:
        paths[condition] = metamer_dir / f"{condition}_{image_name}.tif"
    images = {
        name: po.load_images(path, as_gray=False).double()
        for name, path in paths.items()
    }
    all_images[image_name] = images

    # RGB mean and covariance errors relative to the target
    target_mean, target_covariance = rgb_mean_and_covariance(images["target"])
    for condition, image in images.items():
        mean, covariance = rgb_mean_and_covariance(image)
        # Largest channel mean difference, and covariance error relative to target
        mean_error = (mean - target_mean).abs().max()
        covariance_error = torch.linalg.norm(
            covariance - target_covariance
        ) / torch.linalg.norm(target_covariance)
        color_rows.append(
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

    # Sum of squared errors per PS statistic family, with the target's PCA transform
    metamers = {name: image for name, image in images.items() if name != "target"}
    transform = po.process.PCA(images["target"])
    for row in family_errors(images["target"], metamers, transform):
        family_rows.append({"image": image_name, **row})

write_results(color_rows, results_dir / "color_metrics.csv")
write_results(family_rows, results_dir / "family_errors.csv")

# Print a summary table of the color metrics
print(
    f"{'image':<10} {'condition':<16} {'mean RGB':<20} {'mean err':>8} {'cov err':>8}"
)
for row in color_rows:
    mean_rgb = f"{row['mean_r']:.3f} {row['mean_g']:.3f} {row['mean_b']:.3f}"
    print(
        f"{row['image']:<10} {row['condition']:<16} {mean_rgb:<20} "
        f"{row['mean_error']:>8.4f} {row['covariance_error']:>8.4f}"
    )

# Plot the family errors of each image, with a shared log scale
labels = {name: label for label, _, name, _ in statistics_to_compare}
errors = [row["sse"] for row in family_rows if row["sse"] > 0]
limits = (min(errors) / 2, max(errors) * 2)
for image_name in image_names:
    rows = [row for row in family_rows if row["image"] == image_name]
    plot_errors(
        rows,
        image_name,
        "PCA",
        labels,
        ["matlab", *conditions],
        limits,
        None,
        results_dir / f"family_errors_{image_name}.png",
    )

# Show the target and all metamers, one image per row
columns = ["target", "matlab", *conditions]
fig, axes = plt.subplots(
    len(image_names), len(columns), figsize=(2 * len(columns), 2 * len(image_names))
)
for row_axes, image_name in zip(axes, image_names):
    for ax, name in zip(row_axes, columns):
        image = all_images[image_name][name].squeeze(0).permute(1, 2, 0)
        ax.imshow(image.clamp(0, 1).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
    row_axes[0].set_ylabel(image_name)
for ax, name in zip(axes[0], columns):
    ax.set_title(name, fontsize=9)
fig.savefig(results_dir / "metamers.png", dpi=160, bbox_inches="tight")
