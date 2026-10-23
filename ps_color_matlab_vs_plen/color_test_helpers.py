"""Convert MATLAB PS statistics using shared grayscale and color test helpers.

``construct_normalizing_dict`` and ``remove_redundant_and_normalize`` follow
plenoptic's ``tests/test_models.py``, with branches on ``plen_ps.color_statistics``
for joint channel/orientation layouts and the extra color statistic families.
"""

from collections import OrderedDict

import einops
import torch

import plenoptic as po
from plenoptic.models.portilla_simoncelli import LOWPASS_SHIFTS


def matlab_to_plenoptic_dict_color(params, n_scales, n_orientations):
    """Map a MATLAB color ``params`` dictionary to plenoptic names and layouts.

    ``params`` is loaded with ``scipy.io.loadmat(..., simplify_cells=True)``;
    ``n_scales`` and ``n_orientations`` specify the pyramid layout. Returned tensors
    have a leading singleton batch dimension. Per-channel statistics retain their
    channel axis, while joint statistics combine channel and orientation.

    This only renames, rearranges, and selects active blocks: correlations are
    still covariances, phase signs are unchanged, and symmetric entries remain.
    The normalization helper adds ``std_reconstructed`` and ``magnitude_std``.
    Tensor copies keep later normalization from modifying the MATLAB arrays.
    """
    params = {key: torch.tensor(value) for key, value in params.items()}
    n_channels = params["pixelStats"].shape[0]
    n_signals = n_channels * n_orientations
    n_lowpass_signals = len(LOWPASS_SHIFTS) * n_channels

    # Statistics shared with grayscale, with MATLAB's channel axis moved first.
    rep = OrderedDict()
    rep["pixel_statistics"] = params["pixelStats"][None]
    rep["auto_correlation_magnitude"] = params["autoCorrMag"].permute(4, 0, 1, 3, 2)[
        None
    ]
    rep["skew_reconstructed"] = params["pixelLPStats"][: n_scales + 1].T[None]
    rep["kurtosis_reconstructed"] = params["pixelLPStats"][n_scales + 1 :].T[None]
    rep["auto_correlation_reconstructed"] = params["autoCorrReal"][
        :, :, : n_scales + 1, :
    ].permute(3, 0, 1, 2)[None]
    rep["var_highpass_residual"] = params["varianceHPR"][None, :, None]

    # Color magnitude matrices already combine channel and orientation. Unlike
    # grayscale MATLAB output, the color MATLAB output has no extra placeholder scale
    # to discard for these statistics.
    rep["cross_orientation_correlation_magnitude"] = params["cousinMagCorr"][None]
    rep["cross_scale_correlation_magnitude"] = params["parentMagCorr"][None]
    rep["cross_scale_correlation_real"] = params["parentRealCorr"][
        :n_signals, : 2 * n_signals, : n_scales - 1
    ][None]

    # Color-only marginal statistics: pixelStats uses RGB, these use PCA bands.
    rep["color_covariance"] = params["colorCorr"][None]
    rep["auto_correlation_transformed"] = params["autoCorrReal"][
        :, :, n_scales + 1, :
    ].permute(2, 0, 1)[None]
    rep["skew_transformed"] = params["pixelStatsPCA"][:, 0][None]
    rep["kurtosis_transformed"] = params["pixelStatsPCA"][:, 1][None]
    # PCA minima and maxima are not constrained by MATLAB synthesis or plenoptic.

    # Retain real correlations, which the grayscale conversion discards. The
    # final cousin/parent planes instead describe the shifted lowpass signals.
    rep["cross_orientation_correlation_real"] = params["cousinRealCorr"][
        :n_signals, :n_signals, :n_scales
    ][None]
    rep["cross_correlation_lowpass"] = params["cousinRealCorr"][
        :n_lowpass_signals, :n_lowpass_signals, n_scales
    ][None]
    rep["cross_correlation_coarsest_scale_lowpass"] = params["parentRealCorr"][
        :n_signals, :n_lowpass_signals, n_scales - 1
    ][None]
    return rep


def construct_normalizing_dict(
    plen_ps: po.models.PortillaSimoncelli,
    img: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Construct dictionary to normalize covariances in PS representation.

    The matlab code computes covariances instead of correlations for the
    cross-orientation and cross-scale correlations. Here, we construct the
    tensors required to normalize those covariances to correlations, which are
    the square root of the outer product of the variances of the tensors that
    created them (intermediaries not present in the final PS model output).

    This extends the helper in plenoptic/tests/test_models.py. Grayscale calls
    use the original two arguments. For color, img must already be transformed
    into the PCA basis. Channel and orientation are combined before the same
    einsum calculations. All variances come from the image's pyramid coefficients,
    including the expanded lowpass used by the additional color statistics.

    """
    _, coeffs, _, lowpass = plen_ps._compute_pyr_coeffs(img)
    mags, reals = plen_ps._compute_intermediate_representations(coeffs)
    doub_mags, doub_sep = plen_ps._double_phase_pyr_coeffs(coeffs)
    mags_var = torch.stack([m.var((-2, -1), correction=0) for m in mags], -1)
    reals_var = torch.stack([r.var((-2, -1), correction=0) for r in reals], -1)

    if plen_ps.color_statistics:
        mags_var = einops.rearrange(mags_var, "b c o s -> b 1 (c o) s")
        reals_var = einops.rearrange(reals_var, "b c o s -> b 1 (c o) s")

    normalizing_dict = {}
    com = einops.einsum(mags_var, mags_var, "b c o1 s, b c o2 s -> b c o1 o2 s")
    normalizing_dict["cross_orientation_correlation_magnitude"] = com.pow(0.5)

    if plen_ps.n_scales > 1:
        doub_mags_var = torch.stack(
            [m.var((-2, -1), correction=0) for m in doub_mags], -1
        )
        doub_sep_var = torch.stack(
            [s.var((-2, -1), correction=0) for s in doub_sep], -1
        )
        if plen_ps.color_statistics:
            doub_mags_var = einops.rearrange(doub_mags_var, "b c o s -> b 1 (c o) s")
            # Each parent's real orientations remain before its imaginary ones.
            doub_sep_var = einops.rearrange(doub_sep_var, "b c o s -> b 1 (c o) s")
        csm = einops.einsum(
            mags_var[..., :-1],
            doub_mags_var,
            "b c o1 s, b c o2 s -> b c o1 o2 s",
        )
        normalizing_dict["cross_scale_correlation_magnitude"] = csm.pow(0.5)
        csr = einops.einsum(
            reals_var[..., :-1],
            doub_sep_var,
            "b c o1 s, b c o2 s -> b c o1 o2 s",
        )
        normalizing_dict["cross_scale_correlation_real"] = csr.pow(0.5)
    else:
        normalizing_dict["cross_scale_correlation_magnitude"] = 1
        normalizing_dict["cross_scale_correlation_real"] = 1

    if plen_ps.color_statistics:
        # Color statistics don't have a channel axis in joint matrices.
        for key, value in normalizing_dict.items():
            if isinstance(value, torch.Tensor):
                normalizing_dict[key] = value.squeeze(1)

        # Additional color families use the real coefficients and expanded lowpass.
        # Correlation betwen real coefficients, same scale. Vacher stat (x)
        reals_var = reals_var.squeeze(1)
        cor = einops.einsum(reals_var, reals_var, "b o1 s, b o2 s -> b o1 o2 s")
        normalizing_dict["cross_orientation_correlation_real"] = cor.pow(0.5)
        # Correlation between lowpass residuals, shifted. Vacher stat (xi)
        # MATLAB divides the expanded lowpass by four before computing covariances
        # (colorTextureSynth/textureColorAnalysis.m, line 232).
        # Periodic shifts preserve each channel's variance.
        expanded_lowpass = po.process.expand(lowpass, 2) / 4
        lowpass_var = expanded_lowpass.var((-2, -1), correction=0)
        lowpass_var = einops.repeat(
            lowpass_var, "b c -> b (c shift)", shift=len(LOWPASS_SHIFTS)
        )
        lowpass = einops.einsum(lowpass_var, lowpass_var, "b o1, b o2 -> b o1 o2")
        normalizing_dict["cross_correlation_lowpass"] = lowpass.pow(0.5)
        # Correlation between coarsest scale real coefficients and lowpass image.
        # Vacher stat (xii)
        coarsest_lowpass = einops.einsum(
            reals_var[..., -1], lowpass_var, "b o1, b o2 -> b o1 o2"
        )
        normalizing_dict["cross_correlation_coarsest_scale_lowpass"] = (
            coarsest_lowpass.pow(0.5)
        )

    return normalizing_dict


def remove_redundant_and_normalize(
    matlab_rep: OrderedDict,
    use_true_correlations: bool,
    plen_ps: po.models.PortillaSimoncelli,
    normalizing_dict: dict,
) -> torch.Tensor:
    """Remove redundant stats from dictionary of representation, and normalize
    correlations

    Redundant stats fall in two categories: those that are not included at all
    anymore (e.g., magnitude means, extra zero placeholders), and those that
    are computed automatically (because of how the computation works) and then
    discarded (e.g., symmetries in autocorrelation). This function removes both.

    Additionally, if ``use_true_correlations=False``, we normalize the
    correlations. The matlab PS code did not do so, so that, e.g., the center
    value of the autocorrelations was the corresponding variance (rather than
    being normalized to 1). Originally, we supported both normalized and
    un-normalized, but now we only support the normalized version. Thus, to
    compare with the matlab outputs (and older plenoptic versions), need to
    normalize the auto- and cross-correlations.

    We also grab the center values of auto_correlation_reconstructed and
    create the new statistic std_reconstructed, as this information is
    important.

    Finally, we take the negative of half the cross_scale_correlation_real.
    When doubling the phase, Portilla-Simoncelli accidentally calculated the
    negative of the real component: they called atan2(real, imag), whereas
    atan2 receives the y-value (thus, the imaginary component) first. This
    caused the correlations between the real coefficients on one scale and the
    real coefficients on the next to be negative where they should be positive,
    and vice versa (this doesn't affect the magnitudes or the imaginary
    components on the next scale).

    This extends the helper in plenoptic/tests/test_models.py. Grayscale input
    uses the original legacy dictionary, including placeholder scales. Color
    input comes from matlab_to_plenoptic_dict_color, which already selects the
    active blocks. Color retains same-scale real correlations, applies the phase
    correction to every parent-channel block, and includes the extra families.
    The dictionary is updated in place in both cases.

    """
    # NOT NEEDED FOR COLOR: Color input has already had unused fields and padding
    # removed by its dictionary adapter; only the grayscale input needs removal
    # of unused stats.
    if not plen_ps.color_statistics:
        # Remove those stats that are not included at all.
        matlab_rep.pop("magnitude_means")
        matlab_rep.pop("cross_orientation_correlation_real")

        # Remove the 0 placeholders
        matlab_rep["cross_scale_correlation_magnitude"] = matlab_rep[
            "cross_scale_correlation_magnitude"
        ][..., :-1]
        matlab_rep["cross_orientation_correlation_magnitude"] = matlab_rep[
            "cross_orientation_correlation_magnitude"
        ][..., :-1]
        matlab_rep["cross_scale_correlation_real"] = matlab_rep[
            "cross_scale_correlation_real"
        ][..., : plen_ps.n_orientations, :, :-1]
        # if there are two orientations, there's some more 0 placeholders
        if plen_ps.n_orientations == 2:
            matlab_rep["cross_scale_correlation_real"] = matlab_rep[
                "cross_scale_correlation_real"
            ][..., :-1, :]

    # Correct the real half of each phase-doubled parent-channel block.
    # In the original implementation, a single slice gets all real coefficients,
    # but here real coefficients for different channels of the parents are combined
    # into one dimension and require more specific targeted indexing.
    n_channels = (
        matlab_rep["pixel_statistics"].shape[1] if plen_ps.color_statistics else 1
    )
    for channel in range(n_channels):
        first_real_column = channel * 2 * plen_ps.n_orientations
        last_real_column = first_real_column + plen_ps.n_orientations
        matlab_rep["cross_scale_correlation_real"][
            ..., first_real_column:last_real_column, :
        ] *= -1

    if not use_true_correlations:
        # Create std_reconstructed
        ctr_ind = plen_ps.spatial_corr_width // 2
        var_recon = matlab_rep["auto_correlation_reconstructed"][
            ..., ctr_ind, ctr_ind, :
        ].clone()
        matlab_rep["std_reconstructed"] = var_recon**0.5

        # Normalize the autocorrelations using their center values
        matlab_rep["auto_correlation_reconstructed"] /= var_recon[..., None, None, :]
        acm_ctr = matlab_rep["auto_correlation_magnitude"][
            ..., ctr_ind, ctr_ind, :, :
        ].clone()
        matlab_rep["auto_correlation_magnitude"] /= acm_ctr[..., None, None, :, :]

        # Create magnitude_std
        diag = torch.arange(
            matlab_rep["cross_orientation_correlation_magnitude"].shape[-2]
        )
        var_mags = matlab_rep["cross_orientation_correlation_magnitude"][
            ..., diag, diag, :
        ]
        matlab_rep["magnitude_std"] = var_mags.pow(0.5)
        if plen_ps.color_statistics:
            matlab_rep["magnitude_std"] = matlab_rep["magnitude_std"].unflatten(
                -2, (n_channels, plen_ps.n_orientations)
            )

        # The cross-correlations are normalized by the outer product of the
        # variances of the tensors that create them. We have created these and
        # saved them in normalizing dict, which we use here
        crosscorr_keys = [
            "cross_scale_correlation_real",
            "cross_scale_correlation_magnitude",
            "cross_orientation_correlation_magnitude",
        ]
        if plen_ps.color_statistics:
            # Normalize the PCA transformed image channels autocorrelations
            act_ctr = matlab_rep["auto_correlation_transformed"][
                ..., ctr_ind, ctr_ind
            ].clone()
            matlab_rep["auto_correlation_transformed"] /= act_ctr[..., None, None]
            # Add crosscorrelation keys specific to color model
            crosscorr_keys += [
                "cross_orientation_correlation_real",
                "cross_correlation_lowpass",
                "cross_correlation_coarsest_scale_lowpass",
            ]
        for k in crosscorr_keys:
            matlab_rep[k] = matlab_rep[k] / normalizing_dict[k]

    # Finally, turn dict back into vector, removing redundant stats
    if plen_ps.color_statistics:
        # The adapter and derived statistics were added in a different order.
        matlab_rep = OrderedDict(
            (key, matlab_rep[key]) for key in plen_ps._necessary_stats_dict
        )
    return plen_ps.convert_to_tensor(matlab_rep)
