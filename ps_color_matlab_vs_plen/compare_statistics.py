"""Compare MATLAB and plenoptic color PS statistics and save their differences."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from color_test_helpers import (
    construct_normalizing_dict,
    load_matlab_statistics,
    matlab_to_plenoptic_dict_color,
    remove_redundant_and_normalize,
)
from compute_stats_plenoptic import compute_color_ps_statistics

import plenoptic as po


def compare_statistic_arrays(
    name, matlab_array, plenoptic_array, necessary_mask=None, verbose=True
):
    """Compare corresponding MATLAB and plenoptic statistic arrays.

    Parameters
    ----------
    name : str
        Label printed above the comparison summary.
    matlab_array, plenoptic_array : array-like
        Corresponding statistic arrays with the same shape.
    necessary_mask : array-like or None
        Boolean mask selecting the non-redundant values to summarize. If
        ``None``, all values are included.
    verbose : bool
        Whether to print the comparison summary.

    Returns
    -------
    max_absolute_difference, max_relative_difference : float
        Maximum absolute and relative differences among the selected entries.
    """
    matlab_values = np.asarray(matlab_array)
    plenoptic_values = np.asarray(plenoptic_array)

    if necessary_mask is not None:
        selected = np.broadcast_to(necessary_mask, matlab_values.shape)
        matlab_values = matlab_values[selected]
        plenoptic_values = plenoptic_values[selected]

    absolute_difference = np.abs(matlab_values - plenoptic_values)
    relative_difference = absolute_difference / np.abs(matlab_values)
    max_absolute_difference = absolute_difference.max()
    max_relative_difference = relative_difference.max()

    if verbose:
        print(f"\n{name}")
        print(f"  compared values: {matlab_values.size}")
        print(f"  maximum absolute difference: {max_absolute_difference:.6g}")
        print(f"  mean absolute difference:    {absolute_difference.mean():.6g}")
        print(f"  maximum relative difference: {max_relative_difference:.6g}")
    return max_absolute_difference, max_relative_difference


# Keep the original reporting order and Vacher Appendix B.2 references.
# Each row is (display label, MATLAB field, plenoptic statistic, Vacher statistic).
statistics_to_compare = [
    # Color-specific or joint statistics, and supporting magnitude deviations.
    ("RGB pixel statistics", "pixelStats", "pixel_statistics", "i.c"),
    ("PCA-transformed skewness", "pixelStatsPCA", "skew_transformed", "ix"),
    ("PCA-transformed kurtosis", "pixelStatsPCA", "kurtosis_transformed", "ix"),
    (
        "PCA-image autocorrelation",
        "autoCorrReal",
        "auto_correlation_transformed",
        "viii",
    ),
    ("Pyramid-magnitude standard deviation", "cousinMagCorr", "magnitude_std", "iii-v"),
    (
        "Joint cross-orientation magnitude correlation",
        "cousinMagCorr",
        "cross_orientation_correlation_magnitude",
        "iv",
    ),
    (
        "Joint cross-scale magnitude correlation",
        "parentMagCorr",
        "cross_scale_correlation_magnitude",
        "v",
    ),
    (
        "Joint same-scale real correlation",
        "cousinRealCorr",
        "cross_orientation_correlation_real",
        "x",
    ),
    (
        "Joint cross-scale real/phase correlation",
        "parentRealCorr",
        "cross_scale_correlation_real",
        "vi",
    ),
    ("RGB color covariance", "colorCorr", "color_covariance", "vii"),
    (
        "Shifted-lowpass joint correlation",
        "cousinRealCorr",
        "cross_correlation_lowpass",
        "xi",
    ),
    (
        "Coarsest-real/shifted-lowpass joint correlation",
        "parentRealCorr",
        "cross_correlation_coarsest_scale_lowpass",
        "xii",
    ),
    # Unchanged grayscale families, computed independently per PCA channel.
    ("Highpass-residual variance", "varianceHPR", "var_highpass_residual", "i.b"),
    ("Reconstructed-image skewness", "pixelLPStats", "skew_reconstructed", "i.a"),
    ("Reconstructed-image kurtosis", "pixelLPStats", "kurtosis_reconstructed", "i.a"),
    (
        "Reconstructed-image standard deviation",
        "autoCorrReal",
        "std_reconstructed",
        "ii",
    ),
    (
        "Reconstructed-image autocorrelation",
        "autoCorrReal",
        "auto_correlation_reconstructed",
        "ii",
    ),
    (
        "Pyramid-magnitude autocorrelation",
        "autoCorrMag",
        "auto_correlation_magnitude",
        "iii",
    ),
]


@torch.no_grad()
def compare_color_ps_statistics(matlab, verbose=True):
    """Compare color PS statistic families between MATLAB and plenoptic.

    Compute plenoptic statistics using MATLAB's image, PCA transform, and
    model parameters. Convert MATLAB's statistics to plenoptic's layout, normalize
    covariances to correlations, and compare the non-redundant entries selected
    by the model's masks.

    Parameters
    ----------
    matlab : dict
        Reference loaded with ``load_matlab_statistics``. Contains the image,
        PCA transform and transformed image, model parameters, and MATLAB
        statistics in ``params``.
    verbose : bool
        Whether to print input-image and PCA-image diagnostics and per-family
        error summaries. Diagnostics are not included in the returned rows.

    Returns
    -------
    difference_summaries : list of dict
        One row per family in ``statistics_to_compare``, in the same order.
        Each row contains ``matlab_name``, ``plenoptic_name``, the Appendix B.2
        reference ``vacher_statistic``, and ``max_absolute_difference`` and
        ``max_relative_difference`` over the selected entries. No files are written.
    """
    plenoptic_statistics = compute_color_ps_statistics(matlab)
    plenoptic_representation = plenoptic_statistics["representation"]
    plenoptic_necessary_stats = plenoptic_statistics["necessary_stats"]

    # Input/PCA agreement is a prerequisite for sign-sensitive pyramid comparisons.
    matlab_image = torch.tensor(matlab["im0"], dtype=torch.float64).permute(2, 0, 1)[
        None
    ]
    compare_statistic_arrays(
        "Input image (BCHW, 0--255)",
        matlab_image,
        plenoptic_statistics["image"],
        verbose=verbose,
    )
    matlab_pca_image = torch.tensor(matlab["imPCA"], dtype=torch.float64).permute(
        2, 0, 1
    )[None]
    compare_statistic_arrays(
        "PCA-transformed image",
        matlab_pca_image,
        plenoptic_statistics["pca"]["transformed_image"],
        verbose=verbose,
    )

    # Follow plenoptic's grayscale MATLAB test: prepare a dictionary, construct
    # covariance divisors, then normalize and pack with the model's redundancy mask.
    # The image is already in MATLAB's PCA basis, so do not apply another transform.
    model = po.models.PortillaSimoncelli(
        matlab_pca_image.shape[-2:],
        **plenoptic_statistics["model"],
        color_statistics=True,
        transform=None,
    ).to(torch.float64)
    model.eval()
    po.remove_grad(model)

    matlab_representation = matlab_to_plenoptic_dict_color(
        matlab["params"],
        n_scales=model.n_scales,
        n_orientations=model.n_orientations,
    )
    normalizing_dict = construct_normalizing_dict(model, matlab_pca_image)
    matlab_tensor = remove_redundant_and_normalize(
        matlab_representation, False, model, normalizing_dict
    )
    matlab_representation = model.convert_to_dict(matlab_tensor)

    difference_summaries = []
    for label, matlab_name, plenoptic_name, vacher_statistic in statistics_to_compare:
        max_absolute_difference, max_relative_difference = compare_statistic_arrays(
            label,
            matlab_representation[plenoptic_name],
            plenoptic_representation[plenoptic_name],
            plenoptic_necessary_stats[plenoptic_name],
            verbose=verbose,
        )
        difference_summaries.append(
            {
                "matlab_name": matlab_name,
                "plenoptic_name": plenoptic_name,
                "vacher_statistic": vacher_statistic,
                "max_absolute_difference": max_absolute_difference,
                "max_relative_difference": max_relative_difference,
            }
        )

    return difference_summaries


def write_results(rows, output):
    """Save the per-family comparison summaries to a CSV file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matlab_path",
        nargs="?",
        type=Path,
        default=Path("matlab_statistics/matlab_DSCF4315.mat"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = compare_color_ps_statistics(load_matlab_statistics(args.matlab_path))
    output = args.output or Path("results") / f"{args.matlab_path.stem}_differences.csv"
    write_results(rows, output)
    print(f"\nSaved difference summaries to {output}")
