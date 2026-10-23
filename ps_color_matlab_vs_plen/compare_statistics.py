"""Compare MATLAB and plenoptic color PS statistics and save their differences."""

import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from color_test_helpers import (
    construct_normalizing_dict,
    matlab_to_plenoptic_dict_color,
    remove_redundant_and_normalize,
)

import plenoptic as po


def compare_statistic(name, matlab_array, plenoptic_array, necessary_mask=None):
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

    print(f"\n{name}")
    print(f"  compared values: {matlab_values.size}")
    print(f"  maximum absolute difference: {max_absolute_difference:.6g}")
    print(f"  mean absolute difference:    {absolute_difference.mean():.6g}")
    print(f"  maximum relative difference: {max_relative_difference:.6g}")
    return max_absolute_difference, max_relative_difference


# Load the saved analyses of the same RGB image and MATLAB PCA transform.
image_name = "DSCF4315"
matlab = sio.loadmat(f"matlab_statistics/matlab_{image_name}.mat", simplify_cells=True)
saved_plenoptic = torch.load(
    f"plenoptic_statistics/plenoptic_matlab_pca_{image_name}.pt",
    weights_only=True,
)
plenoptic_representation = saved_plenoptic["representation"]
plenoptic_necessary_stats = saved_plenoptic["necessary_stats"]
np.set_printoptions(precision=6, suppress=True, linewidth=120)

# Input/PCA agreement is a prerequisite for sign-sensitive pyramid comparisons.
matlab_image = torch.tensor(matlab["im0"], dtype=torch.float64).permute(2, 0, 1)[None]
image_max_absolute_difference, image_max_relative_difference = compare_statistic(
    "Input image (BCHW, 0--255)", matlab_image, saved_plenoptic["image"]
)
matlab_pca_image = torch.tensor(matlab["imPCA"], dtype=torch.float64).permute(2, 0, 1)[
    None
]
pca_max_absolute_difference, pca_max_relative_difference = compare_statistic(
    "PCA-transformed image",
    matlab_pca_image,
    saved_plenoptic["pca"]["transformed_image"],
)

# Follow plenoptic's grayscale MATLAB test: prepare a dictionary, construct
# covariance divisors, then normalize and pack with the model's redundancy mask.
# The image is already in MATLAB's PCA basis, so do not apply another transform.
model = po.models.PortillaSimoncelli(
    matlab_pca_image.shape[-2:],
    **saved_plenoptic["model"],
    color_statistics=True,
    transform=None,
).to(torch.float64)
model.eval()
po.remove_grad(model)

with torch.no_grad():
    matlab_representation = matlab_to_plenoptic_dict_color(
        matlab["params"], n_scales=model.n_scales, n_orientations=model.n_orientations
    )
    normalizing_dict = construct_normalizing_dict(model, matlab_pca_image)
    matlab_tensor = remove_redundant_and_normalize(
        matlab_representation, False, model, normalizing_dict
    )
    matlab_representation = model.convert_to_dict(matlab_tensor)

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

difference_summaries = []
for label, matlab_name, plenoptic_name, vacher_statistic in statistics_to_compare:
    max_absolute_difference, max_relative_difference = compare_statistic(
        label,
        matlab_representation[plenoptic_name],
        plenoptic_representation[plenoptic_name],
        plenoptic_necessary_stats[plenoptic_name],
    )
    difference_summaries.append(
        [
            matlab_name,
            plenoptic_name,
            vacher_statistic,
            max_absolute_difference,
            max_relative_difference,
        ]
    )

# Save one summary row for each comparison above
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
results_path = results_dir / "matlab_plen_differences.csv"
with results_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "matlab_name",
            "plenoptic_name",
            "vacher_statistic",
            "max_absolute_difference",
            "max_relative_difference",
        ]
    )
    writer.writerows(difference_summaries)

print(f"\nSaved difference summaries to {results_path}")
