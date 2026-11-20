"""Compare final RGB metamers with their targets under PCA and OPC."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from compare_statistics import statistics_to_compare, write_results

import plenoptic as po


@torch.no_grad()
def family_errors(target, metamers, transform):
    """Return the unweighted sum of squared errors over retained entries per family."""
    model = po.models.PortillaSimoncelli(
        target.shape[-2:],
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=7,
        color_statistics=True,
        transform=transform,
    ).to(torch.float64)
    model.eval()
    target_stats = model(target)
    rows = []
    for condition, metamer in metamers.items():
        if metamer.shape != target.shape:
            raise ValueError(
                f"{condition}: expected {target.shape}, got {metamer.shape}"
            )
        differences = model.convert_to_dict(model(metamer) - target_stats)
        for family, mask in model._necessary_stats_dict.items():
            values = differences[family][..., mask]
            rows.append(
                {
                    "synthesis_condition": condition,
                    "statistic_family": family,
                    "n_entries": values.numel(),
                    "sse": values.square().sum().item(),
                }
            )
    return rows


def plot_errors(rows, image, basis, labels, conditions, limits, linear_threshold, path):
    """Plot one evaluation basis with the same scale and ordering in every figure."""
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    positions = np.arange(len(labels))
    markers = ["o", "s", "^", "D", "v"]
    for index, (condition, marker) in enumerate(zip(conditions, markers, strict=True)):
        errors = {
            row["statistic_family"]: row["sse"]
            for row in rows
            if row["synthesis_condition"] == condition
        }
        ax.scatter(
            [errors[name] for name in labels],
            positions + (index - 2) * 0.065,
            label=condition,
            marker=marker,
            s=24,
            zorder=3,
        )
    if linear_threshold is None:
        ax.set_xscale("log")
    else:
        ax.set_xscale("symlog", linthresh=linear_threshold)
    ax.set_xlim(limits)
    ax.set_yticks(positions, labels.values())
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="both", alpha=0.2)
    ax.set_xlabel("Sum of squared errors")
    ax.set_title(f"{image}   ·   {basis} evaluation", loc="left", weight="bold")
    ax.legend(
        title="Synthesis condition",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(2)
    image_names = json.loads(Path("sweep.json").read_text())["images"]
    output = Path("validation")
    plot_dir = output / "metamer_errors"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in image_names:
        target = po.load_images(Path("inputs") / f"{image}.tif", as_gray=False).double()
        paths = {"MATLAB": Path("matlab_metamers") / f"matlab_{image}.tif"}
        for space in ("pca", "opc"):
            for init in ("default", "color_matched"):
                condition = f"{space.upper()}_{init}"
                paths[condition] = (
                    Path("plenoptic_metamers") / f"plenoptic_{space}_{init}_{image}.tif"
                )
        metamers = {
            name: po.load_images(path, as_gray=False).double()
            for name, path in paths.items()
        }
        for basis, transform in (
            ("PCA", po.process.PCA(target)),
            ("OPC", po.process.OPC()),
        ):
            for row in family_errors(target, metamers, transform):
                rows.append({"image": image, "evaluation_transform": basis, **row})

    write_results(rows, output / "metamer_errors.csv")
    # Use a linear range below the smallest positive SSE to show exact zeros.
    scores = np.array([row["sse"] for row in rows])
    positive = scores[scores > 0]
    low = 10.0 ** np.floor(np.log10(positive.min())) if positive.size else 1.0
    high = (
        max(10.0 ** np.ceil(np.log10(positive.max())), 10 * low)
        if positive.size
        else 1.0
    )
    linear_threshold = low if (scores == 0).any() else None
    limits = (-low / 10 if linear_threshold else low / 1.2, high * 1.2)
    conditions = list(dict.fromkeys(row["synthesis_condition"] for row in rows))
    for image in image_names:
        for basis in ("PCA", "OPC"):
            labels = {
                name: label.replace("PCA", basis)
                for label, _, name, _ in statistics_to_compare
            }
            selected = [
                r
                for r in rows
                if r["image"] == image and r["evaluation_transform"] == basis
            ]
            plot_errors(
                selected,
                image,
                basis,
                labels,
                conditions,
                limits,
                linear_threshold,
                plot_dir / f"{image}_{basis.lower()}.png",
            )
