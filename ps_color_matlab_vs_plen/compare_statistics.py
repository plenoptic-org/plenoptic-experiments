import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

import plenoptic as po

# The summaries use plenoptic's saved necessary-statistic masks. MATLAB retains
# symmetric and otherwise redundant entries that plenoptic intentionally drops.

image_name = "DSCF4315"
matlab = sio.loadmat(f"matlab_statistics/matlab_{image_name}.mat", simplify_cells=True)
saved_plenoptic = torch.load(
    f"plenoptic_statistics/plenoptic_matlab_pca_{image_name}.pt",
    weights_only=True,
)

matlab_representation = matlab["params"]

# This is the actual OrderedDict returned by model.convert_to_dict(). Its values
# are torch tensors with informative shapes and NaNs at redundant positions.
plenoptic_representation = saved_plenoptic["representation"]
plenoptic_necessary_stats = saved_plenoptic["necessary_stats"]

n_scales = saved_plenoptic["model"]["n_scales"]
n_orientations = saved_plenoptic["model"]["n_orientations"]
spatial_corr_width = saved_plenoptic["model"]["spatial_corr_width"]
n_channels = 3
center = spatial_corr_width // 2

np.set_printoptions(precision=6, suppress=True, linewidth=120)
difference_summaries = []


def compare_statistic(name, matlab_array, plenoptic_array, necessary_mask=None):
    """Compare corresponding MATLAB and plenoptic statistic arrays.

    Both arrays are converted to NumPy arrays and subtracted as MATLAB minus
    plenoptic. The printed summary includes only entries that are finite in
    both arrays and, when supplied, selected by ``necessary_mask``. The mask is
    broadcast to the shape of the difference array, matching how plenoptic's
    saved necessary-statistic masks apply across batch dimensions.

    Parameters
    ----------
    name : str
        Label printed above the comparison summary.
    matlab_array, plenoptic_array : array-like
        Corresponding statistic arrays with broadcast-compatible shapes.
    necessary_mask : array-like or None
        Boolean mask selecting the non-redundant values to summarize. If
        ``None``, all values that are finite in both arrays are included.

    Returns
    -------
    max_absolute_difference, max_relative_difference : float
        Maximum absolute and relative differences among the selected entries.
    """
    matlab_array = np.asarray(matlab_array)
    plenoptic_array = np.asarray(plenoptic_array)
    difference = matlab_array - plenoptic_array

    selected = np.isfinite(matlab_array) & np.isfinite(plenoptic_array)
    if necessary_mask is not None:
        selected &= np.broadcast_to(necessary_mask, difference.shape)

    matlab_values = matlab_array[selected]
    plenoptic_values = plenoptic_array[selected]
    absolute_difference = np.abs(difference[selected])
    relative_scale = np.maximum(np.abs(matlab_values), np.abs(plenoptic_values))
    relative_difference = absolute_difference / np.maximum(
        relative_scale, np.finfo(float).eps
    )
    max_absolute_difference = absolute_difference.max()
    max_relative_difference = relative_difference.max()

    print(f"\n{name}")
    print(f"  compared values: {selected.sum()}")
    print(f"  maximum absolute difference: {max_absolute_difference:.6g}")
    print(f"  mean absolute difference:    {absolute_difference.mean():.6g}")
    print(f"  maximum relative difference: {max_relative_difference:.6g}")
    return max_absolute_difference, max_relative_difference


# =============================================================================
# 0. Shared input and PCA diagnostics
# Not a Vacher statistic; prerequisite for the comparisons below
# =============================================================================
# These are prerequisites, not texture statistics. If the input or PCA image
# differs, sign-sensitive pyramid statistics cannot be compared meaningfully.

plenoptic_stat_name = "image"
matlab_stat_name = "im0"

matlab_image = np.transpose(matlab[matlab_stat_name], (2, 0, 1))[None].astype(
    np.float64
)
plenoptic_image = saved_plenoptic[plenoptic_stat_name]
image_max_absolute_difference, image_max_relative_difference = compare_statistic(
    "Input image (BCHW, 0--255)",
    matlab_image,
    plenoptic_image,
)

plenoptic_stat_name = "transformed_image"
matlab_stat_name = "imPCA"
matlab_pca_image = np.transpose(matlab[matlab_stat_name], (2, 0, 1))[None]
plenoptic_pca_mean = saved_plenoptic["pca"]["mean"]
plenoptic_pca_matrix = saved_plenoptic["pca"]["matrix"]
plenoptic_pca_image = saved_plenoptic["pca"][plenoptic_stat_name]

pca_max_absolute_difference, pca_max_relative_difference = compare_statistic(
    "PCA-transformed image",
    matlab_pca_image,
    plenoptic_pca_image,
)

# =============================================================================
# COLOR-SPECIFIC OR COLOR-CHANGED STATISTICS
# =============================================================================
# This first part contains statistics added by the color model, statistics that
# become joint across transformed channels, and supporting variances needed to
# convert MATLAB covariances into plenoptic's normalized correlations. The RGB
# marginal statistics are also kept here because Vacher specifies that they are
# computed before the color transform.

# =============================================================================
# 1. RGB pixel statistics: mean, variance, skew, kurtosis, min, max
# Vacher statistic (i.c), color-model exception
# =============================================================================
# MATLAB and plenoptic both use sample variance (N-1) in this statistic.

plenoptic_stat_name = "pixel_statistics"
matlab_stat_name = "pixelStats"

matlab_pixel_statistics = matlab_representation[matlab_stat_name][None]
plenoptic_pixel_statistics = plenoptic_representation[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "RGB pixel statistics",
    matlab_pixel_statistics,
    plenoptic_pixel_statistics,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "i.c",
        max_absolute_difference,
        max_relative_difference,
    ]
)

# Inspect one column at a time when a mismatch appears:
# matlab_pixel_statistics[..., 0]  # mean
# matlab_pixel_statistics[..., 1]  # sample variance
# matlab_pixel_statistics[..., 2]  # skewness
# matlab_pixel_statistics[..., 3]  # kurtosis
# matlab_pixel_statistics[..., 4]  # minimum
# matlab_pixel_statistics[..., 5]  # maximum


# =============================================================================
# 2. PCA-transformed image skewness
# Vacher statistic (ix)
# =============================================================================

plenoptic_stat_name = "skew_transformed"
matlab_stat_name = "pixelStatsPCA"

matlab_skew_transformed = matlab_representation[matlab_stat_name][:, 0][None]
plenoptic_skew_transformed = plenoptic_representation[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "PCA-transformed skewness",
    matlab_skew_transformed,
    plenoptic_skew_transformed,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "ix",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 3. PCA-transformed image kurtosis
# Vacher statistic (ix)
# =============================================================================

plenoptic_stat_name = "kurtosis_transformed"
matlab_stat_name = "pixelStatsPCA"

matlab_kurtosis_transformed = matlab_representation[matlab_stat_name][:, 1][None]
plenoptic_kurtosis_transformed = plenoptic_representation[plenoptic_stat_name]
kurtosis_transformed_mask = plenoptic_necessary_stats[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "PCA-transformed kurtosis",
    matlab_kurtosis_transformed,
    plenoptic_kurtosis_transformed,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "ix",
        max_absolute_difference,
        max_relative_difference,
    ]
)

# MATLAB pixelStatsPCA columns 3 and 4 contain PCA minima and maxima. Plenoptic
# intentionally has no corresponding public statistic because MATLAB computes
# but does not impose those values during synthesis. They can still be compared
# with plenoptic_pca_image.min/max as a preprocessing diagnostic.


# =============================================================================
# 4. Autocorrelation of the PCA-transformed image channels
# Vacher statistic (viii)
# =============================================================================
# The final autoCorrReal plane is the autocorrelation of each PCA channel.

plenoptic_stat_name = "auto_correlation_transformed"
matlab_stat_name = "autoCorrReal"

matlab_auto_correlation_transformed_raw = np.transpose(
    matlab_representation[matlab_stat_name][:, :, n_scales + 1, :], (2, 0, 1)
)[None]
matlab_variance_transformed = matlab_auto_correlation_transformed_raw[
    :, :, center, center
]
# Note: We normalize this for consistency, but this autocorrelation is
# already the same as the normalized version, because this is computed on
# whitened PCA images in the Matlab code.
matlab_auto_correlation_transformed = (
    matlab_auto_correlation_transformed_raw
    / matlab_variance_transformed[:, :, None, None]
)

plenoptic_auto_correlation_transformed = plenoptic_representation[plenoptic_stat_name]

auto_correlation_transformed_mask = plenoptic_necessary_stats[plenoptic_stat_name]
max_absolute_difference, max_relative_difference = compare_statistic(
    "PCA-image autocorrelation",
    matlab_auto_correlation_transformed,
    plenoptic_auto_correlation_transformed,
    auto_correlation_transformed_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "viii",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 5. Magnitude standard deviation needed for correlation normalization
# Supporting quantity for Vacher statistics (iii)--(v)
# =============================================================================
# MATLAB's cousinMagCorr diagonal contains magnitude variances. In plenoptic they
# are saved into their own statistic.
# The standard deviations are ordered as channel * n_orientations + orientation,
# matching plenoptic after the (channel, orientation) reshape.
#
# These standard deviations will also be used to normalize other statistics
# to be correlations instead of covariances.

plenoptic_stat_name = "magnitude_std"
matlab_stat_name = "cousinMagCorr"

matlab_cousin_magnitude_covariance = matlab_representation[matlab_stat_name]
matlab_magnitude_variance = np.stack(
    [
        np.diag(matlab_cousin_magnitude_covariance[:, :, scale])
        for scale in range(n_scales)
    ],
    axis=-1,
)
matlab_magnitude_std = np.sqrt(
    matlab_magnitude_variance.reshape(n_channels, n_orientations, n_scales)
)[None]

plenoptic_magnitude_std = plenoptic_representation[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Pyramid-magnitude standard deviation",
    matlab_magnitude_std,
    plenoptic_magnitude_std,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "iii-v",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 6. Joint cross-orientation magnitude correlation
# Vacher statistic (iv), made joint across transformed channels
# =============================================================================

plenoptic_stat_name = "cross_orientation_correlation_magnitude"
matlab_stat_name = "cousinMagCorr"

matlab_cousin_magnitude_covariance = matlab_representation[matlab_stat_name]

# MATLAB stores an unnormalized 12 x 12 covariance at every scale. Normalize
# it by the square root of the outer product of its diagonal variances to
# obtain correlations we can compare to plenoptic
matlab_cross_orientation_magnitude_denominator = np.sqrt(
    matlab_magnitude_variance[:, None, :] * matlab_magnitude_variance[None, :, :]
)
matlab_cross_orientation_correlation_magnitude = (
    matlab_cousin_magnitude_covariance / matlab_cross_orientation_magnitude_denominator
)[None]

plenoptic_cross_orientation_correlation_magnitude = plenoptic_representation[
    plenoptic_stat_name
]
cross_orientation_correlation_magnitude_mask = plenoptic_necessary_stats[
    plenoptic_stat_name
]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Joint cross-orientation magnitude correlation",
    matlab_cross_orientation_correlation_magnitude,
    plenoptic_cross_orientation_correlation_magnitude,
    cross_orientation_correlation_magnitude_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "iv",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 7. Joint cross-scale magnitude correlation
# Vacher statistic (v), made joint across transformed channels
# =============================================================================
# parentMagCorr is a covariance between current-scale demeaned magnitudes and
# next-coarser expanded, demeaned magnitudes. The parent
# variances are not stored in the MATLAB params structure, so we need to
# obtain them by reconstructing the expanded demeaned magnitude bands.

plenoptic_stat_name = "cross_scale_correlation_magnitude"
matlab_stat_name = "parentMagCorr"

# We will use matlab pca-transformed image to get the magnitudes
# Because of that, we set the model transform to None (no transform)
torch_pca_image = torch.as_tensor(matlab_pca_image, dtype=torch.float64)
model = po.models.PortillaSimoncelli(
    torch_pca_image.shape[-2:],
    n_scales=n_scales,
    n_orientations=n_orientations,
    spatial_corr_width=spatial_corr_width,
    color_statistics=True,
    transform=None,
).to(torch.float64)
model.eval()
po.remove_grad(model)

with torch.no_grad():
    pyramid_coefficients = model._compute_pyr_coeffs(torch_pca_image)[1]
    magnitude_coefficients, real_coefficients = (
        model._compute_intermediate_representations(pyramid_coefficients)
    )
    # scale up and double phase of pyramid coefficients, to compute their variance
    parent_magnitudes, phase_doubled_real_imag = model._double_phase_pyr_coeffs(
        pyramid_coefficients
    )

# Compute the variance of each channel/orientation/scale
magnitude_variance = torch.stack(
    [coeff.var((-2, -1), correction=0) for coeff in magnitude_coefficients], -1
)
parent_magnitude_variance = torch.stack(
    [coeff.var((-2, -1), correction=0) for coeff in parent_magnitudes],
    -1,
)
# Combine color and orientation channels
magnitude_variance_joint = magnitude_variance.flatten(1, 2).numpy()
parent_magnitude_variance_joint = parent_magnitude_variance.flatten(1, 2).numpy()

# Compute the normalization factors to turn into correlation
matlab_cross_scale_magnitude_denominator = np.sqrt(
    magnitude_variance_joint[:, :, None, :-1]  # Discard last scale
    * parent_magnitude_variance_joint[:, None, :, :]
)
matlab_cross_scale_correlation_magnitude = (
    matlab_representation[matlab_stat_name][None]
    / matlab_cross_scale_magnitude_denominator
)

plenoptic_cross_scale_correlation_magnitude = plenoptic_representation[
    plenoptic_stat_name
]
cross_scale_correlation_magnitude_mask = plenoptic_necessary_stats[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Joint cross-scale magnitude correlation",
    matlab_cross_scale_correlation_magnitude,
    plenoptic_cross_scale_correlation_magnitude,
    cross_scale_correlation_magnitude_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "v",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 8. Joint same-scale real correlation
# Vacher statistic (x)
# =============================================================================
# The first n_scales planes of cousinRealCorr contain 12 x 12 same-scale real
# covariances. Normalize each plane using its own diagonal.

plenoptic_stat_name = "cross_orientation_correlation_real"
matlab_stat_name = "cousinRealCorr"

matlab_cousin_real_covariance = matlab_representation[matlab_stat_name]
# Select the non-empty parameters
matlab_same_scale_real_covariance = matlab_cousin_real_covariance[
    : n_channels * n_orientations,
    : n_channels * n_orientations,
    :n_scales,
]

# For this statistic, the variances are in the diagonal of the matrices
matlab_real_variance_joint = np.stack(
    [
        np.diag(matlab_same_scale_real_covariance[:, :, scale])
        for scale in range(n_scales)
    ],
    axis=-1,
)

# Compute the normalization factors to turn covariances into correlations
matlab_cross_orientation_real_denominator = np.sqrt(
    matlab_real_variance_joint[:, None, :] * matlab_real_variance_joint[None, :, :]
)
matlab_cross_orientation_correlation_real = (
    matlab_same_scale_real_covariance / matlab_cross_orientation_real_denominator
)[None]

plenoptic_cross_orientation_correlation_real = plenoptic_representation[
    plenoptic_stat_name
]
cross_orientation_correlation_real_mask = plenoptic_necessary_stats[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Joint same-scale real correlation",
    matlab_cross_orientation_correlation_real,
    plenoptic_cross_orientation_correlation_real,
    cross_orientation_correlation_real_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "x",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 9. Joint cross-scale real/phase correlation
# Vacher statistic (vi), made joint across transformed channels
# =============================================================================
# MATLAB parentRealCorr uses the opposite sign for the real half of every
# phase-doubled parent-channel block because its atan2 arguments are reversed.
# Correct those columns before normalization. As in section 7, the parent
# second moments come from plenoptic intermediates and should eventually be
# exported by MATLAB for a fully independent comparison.

plenoptic_stat_name = "cross_scale_correlation_real"
matlab_stat_name = "parentRealCorr"

matlab_parent_real_covariance = matlab_representation[matlab_stat_name][
    :, :, : n_scales - 1
].copy()
for channel in range(n_channels):
    # Find indices of columns corresponding to real coefficients
    first_real_column = channel * 2 * n_orientations
    last_real_column = first_real_column + n_orientations
    # Switch sign, because Matlab code uses atan wrong
    # This is a known issue, mentioned in tests/test_models.py:616
    matlab_parent_real_covariance[:, first_real_column:last_real_column, :] *= -1

# Get variance of real coefficients
real_variance = torch.stack(
    [coeff.var((-2, -1), correction=0) for coeff in real_coefficients], -1
)
# Get variance of expanded coefficients
phase_doubled_real_imag_variance = torch.stack(
    [coeff.var((-2, -1), correction=0) for coeff in phase_doubled_real_imag], -1
)
real_variance_joint = real_variance.flatten(1, 2).numpy()
phase_doubled_real_imag_variance_joint = phase_doubled_real_imag_variance.flatten(
    1, 2
).numpy()
# Combine variances into the denominator to convert cov into correlations
matlab_cross_scale_real_denominator = np.sqrt(
    real_variance_joint[:, :, None, :-1]
    * phase_doubled_real_imag_variance_joint[:, None, :, :]
)
# Compute the matlab correlations
matlab_cross_scale_correlation_real = (
    matlab_parent_real_covariance[None] / matlab_cross_scale_real_denominator
)

plenoptic_cross_scale_correlation_real = plenoptic_representation[plenoptic_stat_name]
cross_scale_correlation_real_mask = plenoptic_necessary_stats[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Joint cross-scale real/phase correlation",
    matlab_cross_scale_correlation_real,
    plenoptic_cross_scale_correlation_real,
    cross_scale_correlation_real_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "vi",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 10. RGB color covariance
# Vacher statistic (vii)
# =============================================================================
# Both are population covariances (N denominator). Plenoptic's public mask keeps
# only the strict lower triangle because RGB variances already occur in the
# pixel-statistics family.

plenoptic_stat_name = "color_covariance"
matlab_stat_name = "colorCorr"

matlab_color_covariance = matlab_representation[matlab_stat_name][None]
plenoptic_color_covariance = plenoptic_representation[plenoptic_stat_name]
color_covariance_mask = plenoptic_necessary_stats[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "RGB color covariance",
    matlab_color_covariance,
    plenoptic_color_covariance,
    color_covariance_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "vii",
        max_absolute_difference,
        max_relative_difference,
    ]
)

# =============================================================================
# 11. Shifted-lowpass joint correlation
# Vacher statistic (xi)
# =============================================================================
# Select MATLAB's active 15 x 15 block, then normalize using its diagonal.

plenoptic_stat_name = "cross_correlation_lowpass"
matlab_stat_name = "cousinRealCorr"

matlab_cousin_real_covariance = matlab_representation[matlab_stat_name]
matlab_lowpass_covariance = matlab_cousin_real_covariance[
    : 5 * n_channels, : 5 * n_channels, n_scales
]
matlab_lowpass_variance = np.diag(matlab_lowpass_covariance)
matlab_lowpass_denominator = np.sqrt(
    matlab_lowpass_variance[:, None] * matlab_lowpass_variance[None, :]
)
matlab_cross_correlation_lowpass = (
    matlab_lowpass_covariance / matlab_lowpass_denominator
)[None]

plenoptic_cross_correlation_lowpass = plenoptic_representation[plenoptic_stat_name]
cross_correlation_lowpass_mask = plenoptic_necessary_stats[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Shifted-lowpass joint correlation",
    matlab_cross_correlation_lowpass,
    plenoptic_cross_correlation_lowpass,
    cross_correlation_lowpass_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "xi",
        max_absolute_difference,
        max_relative_difference,
    ]
)

# =============================================================================
# 12. Coarsest-real/shifted-lowpass joint correlation
# Vacher statistic (xii)
# =============================================================================
# The final parentRealCorr plane correlates the 12 coarsest real coefficients
# with the 15 shifted lowpass signals in the same order used by plenoptic. Its
# row and column variances are available from the diagonals already exposed in
# sections 8 and 11, so this normalization is independent of plenoptic.

# Note: The maximum relative difference is large, because some correlations
# are almost 0. I think this is because of the maths: the lowpass residual of a
# given band is by definition uncorrelated to the coarsest pyramid output.
# The maximum absolute difference is ~10^-9


plenoptic_stat_name = "cross_correlation_coarsest_scale_lowpass"
matlab_stat_name = "parentRealCorr"

# Get the covariance to normalize
matlab_coarsest_lowpass_covariance = matlab_representation[matlab_stat_name][
    :, : 5 * n_channels, n_scales - 1
]
matlab_coarsest_real_variance = matlab_real_variance_joint[:, n_scales - 1]
matlab_coarsest_lowpass_denominator = np.sqrt(
    matlab_coarsest_real_variance[:, None] * matlab_lowpass_variance[None, :]
)
matlab_cross_correlation_coarsest_scale_lowpass = (
    matlab_coarsest_lowpass_covariance / matlab_coarsest_lowpass_denominator
)[None]

plenoptic_cross_correlation_coarsest_scale_lowpass = plenoptic_representation[
    plenoptic_stat_name
]
cross_correlation_coarsest_scale_lowpass_mask = plenoptic_necessary_stats[
    plenoptic_stat_name
]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Coarsest-real/shifted-lowpass joint correlation",
    matlab_cross_correlation_coarsest_scale_lowpass,
    plenoptic_cross_correlation_coarsest_scale_lowpass,
    cross_correlation_coarsest_scale_lowpass_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "xii",
        max_absolute_difference,
        max_relative_difference,
    ]
)

# =============================================================================
# STATISTIC FAMILIES BELOW ARE THE SAME AS IN THE ORIGINAL GRAYSCALE MODEL
# =============================================================================
# In the color model these families are evaluated independently on each
# transformed channel. Their definitions are otherwise unchanged.


# =============================================================================
# 13. Highpass-residual variance
# Vacher statistic (i.b), unchanged per-channel family
# =============================================================================

plenoptic_stat_name = "var_highpass_residual"
matlab_stat_name = "varianceHPR"

matlab_var_highpass_residual = matlab_representation[matlab_stat_name][None, :, None]
plenoptic_var_highpass_residual = plenoptic_representation[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Highpass-residual variance",
    matlab_var_highpass_residual,
    plenoptic_var_highpass_residual,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "i.b",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 14. Reconstructed-image multiscale skewness
# Vacher statistic (i.a), unchanged per-channel family
# =============================================================================
# MATLAB stores scale first and color second. Plenoptic stores color first and
# uses fine-to-coarse scale order, which is also MATLAB's order here.

plenoptic_stat_name = "skew_reconstructed"
matlab_stat_name = "pixelLPStats"

pixel_lowpass_statistics = matlab_representation[matlab_stat_name]
# Skewness and kurtosis are in the same MATLAB array; extract skewness.
matlab_skew_reconstructed = pixel_lowpass_statistics[: n_scales + 1].T[None]
plenoptic_skew_reconstructed = plenoptic_representation[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Reconstructed-image skewness",
    matlab_skew_reconstructed,
    plenoptic_skew_reconstructed,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "i.a",
        max_absolute_difference,
        max_relative_difference,
    ]
)

# If only a low-energy band differs, inspect whether MATLAB emitted 0 while
# plenoptic did not. Their low-energy threshold denominators are not identical.


# =============================================================================
# 15. Reconstructed-image multiscale kurtosis
# Vacher statistic (i.a), unchanged per-channel family
# =============================================================================

plenoptic_stat_name = "kurtosis_reconstructed"
matlab_stat_name = "pixelLPStats"

# Skewness and kurtosis are in the same MATLAB array; extract kurtosis.
pixel_lowpass_statistics = matlab_representation[matlab_stat_name]
matlab_kurtosis_reconstructed = pixel_lowpass_statistics[n_scales + 1 :].T[None]
plenoptic_kurtosis_reconstructed = plenoptic_representation[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Reconstructed-image kurtosis",
    matlab_kurtosis_reconstructed,
    plenoptic_kurtosis_reconstructed,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "i.a",
        max_absolute_difference,
        max_relative_difference,
    ]
)

# The analogous low-energy default is kurtosis 3.


# =============================================================================
# 16. Reconstructed-image standard deviation
# Supporting quantity for Vacher statistic (ii), unchanged per-channel family
# =============================================================================
# MATLAB stores the variance at the center of each reconstructed-image
# autocorrelation. Plenoptic exposes the corresponding standard deviation as a
# separate statistic.

plenoptic_stat_name = "std_reconstructed"
matlab_stat_name = "autoCorrReal"

matlab_auto_correlation_real = matlab_representation[matlab_stat_name]
matlab_auto_correlation_reconstructed_raw = np.transpose(
    matlab_auto_correlation_real[:, :, : n_scales + 1, :], (3, 0, 1, 2)
)[None]
matlab_variance_reconstructed = matlab_auto_correlation_reconstructed_raw[
    :, :, center, center, :
]
matlab_std_reconstructed = np.sqrt(matlab_variance_reconstructed)
plenoptic_std_reconstructed = plenoptic_representation[plenoptic_stat_name]
std_reconstructed_mask = plenoptic_necessary_stats[plenoptic_stat_name]

max_absolute_difference, max_relative_difference = compare_statistic(
    "Reconstructed-image standard deviation",
    matlab_std_reconstructed,
    plenoptic_std_reconstructed,
    std_reconstructed_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "ii",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 17. Reconstructed-image normalized autocorrelation
# Vacher statistic (ii), unchanged per-channel family
# =============================================================================
# MATLAB stores covariance-like autocorrelations whose center is the variance.
# Divide each spatial matrix by its center to match plenoptic's normalized form.

plenoptic_stat_name = "auto_correlation_reconstructed"
matlab_stat_name = "autoCorrReal"
matlab_auto_correlation_real = matlab_representation[matlab_stat_name]
matlab_auto_correlation_reconstructed_raw = np.transpose(
    matlab_auto_correlation_real[:, :, : n_scales + 1, :], (3, 0, 1, 2)
)[None]
matlab_variance_reconstructed = matlab_auto_correlation_reconstructed_raw[
    :, :, center, center, :
]
matlab_auto_correlation_reconstructed = (
    matlab_auto_correlation_reconstructed_raw
    / matlab_variance_reconstructed[:, :, None, None, :]
)
plenoptic_auto_correlation_reconstructed = plenoptic_representation[plenoptic_stat_name]
auto_correlation_reconstructed_mask = plenoptic_necessary_stats[plenoptic_stat_name]
max_absolute_difference, max_relative_difference = compare_statistic(
    "Reconstructed-image autocorrelation",
    matlab_auto_correlation_reconstructed,
    plenoptic_auto_correlation_reconstructed,
    auto_correlation_reconstructed_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "ii",
        max_absolute_difference,
        max_relative_difference,
    ]
)


# =============================================================================
# 18. Magnitude normalized autocorrelation
# Vacher statistic (iii), unchanged per-channel family
# =============================================================================
# MATLAB axes are (lag_y, lag_x, scale, orientation, channel). Reorder them to
# plenoptic's (channel, lag_y, lag_x, orientation, scale), then normalize each
# spatial matrix by its center variance.

plenoptic_stat_name = "auto_correlation_magnitude"
matlab_stat_name = "autoCorrMag"
matlab_auto_correlation_magnitude_raw = np.transpose(
    matlab_representation[matlab_stat_name], (4, 0, 1, 3, 2)
)[None]
matlab_auto_correlation_magnitude_center = matlab_auto_correlation_magnitude_raw[
    :, :, center, center, :, :
]
matlab_auto_correlation_magnitude = (
    matlab_auto_correlation_magnitude_raw
    / matlab_auto_correlation_magnitude_center[:, :, None, None, :, :]
)
plenoptic_auto_correlation_magnitude = plenoptic_representation[plenoptic_stat_name]
auto_correlation_magnitude_mask = plenoptic_necessary_stats[plenoptic_stat_name]
max_absolute_difference, max_relative_difference = compare_statistic(
    "Pyramid-magnitude autocorrelation",
    matlab_auto_correlation_magnitude,
    plenoptic_auto_correlation_magnitude,
    auto_correlation_magnitude_mask,
)

difference_summaries.append(
    [
        matlab_stat_name,
        plenoptic_stat_name,
        "iii",
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
