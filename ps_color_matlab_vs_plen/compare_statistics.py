"""Compare MATLAB and plenoptic color PS statistics and save their differences."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from matlab_statistics_helpers import (
    construct_normalizing_dict,
    load_matlab_statistics,
    matlab_to_plenoptic_dict_color,
    remove_redundant_and_normalize,
)

import plenoptic as po


@torch.no_grad()
def compute_color_ps_statistics(matlab):
    """Compute color PS statistics from the image and PCA parameters exported by MATLAB.

    PCA has multiple equivalent solutions, including arbitrary eigenvector signs.
    Reusing MATLAB's transform ensures both implementations use the same PCA
    basis for comparison. Computation uses float64 with gradients disabled.

    Parameters
    ----------
    matlab : dict
        Reference loaded with ``load_matlab_statistics``. Must contain the RGB
        image ``im0``, transformed image ``imPCA``, PCA parameters ``pcaMean`` and
        ``pcaMatrix``, and model parameters ``Nsc`` (scales), ``Nor`` (orientations),
        and ``Na`` (spatial correlation width).

    Returns
    -------
    statistics : dict
        Contains the input ``image``, ``pca`` transform parameters and transformed
        images (including their maximum absolute difference), and ``model``
        constructor parameters. Statistics are returned as a per-family
        ``representation`` dictionary.
        ``necessary_stats`` contains the model's masks for non-redundant entries.
        No files are written.

    Raises
    ------
    AssertionError
        If the computed PCA-transformed image differs from MATLAB's ``imPCA``
        beyond ``rtol=1e-10`` and ``atol=1e-10``.
    """
    image = torch.tensor(matlab["im0"], dtype=torch.float64).permute(2, 0, 1)[None]
    matlab_transformed_image = torch.tensor(
        matlab["imPCA"], dtype=torch.float64
    ).permute(2, 0, 1)[None]

    # Use MATLAB's PCA parameters, since PCA can have different equivalent
    # solutions in MATLAB/Python since eigenvector sign is arbitrary
    pca_matlab = po.process.PCA(image, max_relative_scaling=float("inf"))
    pca_matlab.mean.copy_(
        torch.tensor(matlab["pcaMean"], dtype=torch.float64).reshape(3, 1, 1)
    )
    pca_matlab.matrix.copy_(torch.tensor(matlab["pcaMatrix"], dtype=torch.float64))

    model_parameters = {
        "n_scales": int(matlab["Nsc"]),
        "n_orientations": int(matlab["Nor"]),
        "spatial_corr_width": int(matlab["Na"]),
    }
    model = po.models.PortillaSimoncelli(
        image.shape[-2:],
        **model_parameters,
        color_statistics=True,
        transform=pca_matlab,
    ).to(dtype=torch.float64)
    model.eval()
    po.remove_grad(model)

    representation_tensor = model(image)
    transformed_image = pca_matlab(image)
    torch.testing.assert_close(
        transformed_image,
        matlab_transformed_image,
        rtol=1e-10,
        atol=1e-10,
    )
    return {
        "image": image,
        "pca": {
            "mean": pca_matlab.mean,
            "matrix": pca_matlab.matrix,
            "transformed_image": transformed_image,
            "matlab_transformed_image": matlab_transformed_image,
            "max_abs_difference": (transformed_image - matlab_transformed_image)
            .abs()
            .max(),
        },
        "model": model_parameters,
        "representation": model.convert_to_dict(representation_tensor),
        "necessary_stats": model._necessary_stats_dict,
    }


def compare_statistic_arrays(
    name, matlab_array, plenoptic_array, necessary_mask=None, atol=1e-4, verbose=True
):
    """Compare corresponding MATLAB and plenoptic statistic arrays. Return
    the difference between the arrays, and a flag with values
    "passed" or "failed" indicating if the two match.

    Parameters
    ----------
    name : str
        Label printed above the comparison summary.
    matlab_array, plenoptic_array : array-like
        Corresponding statistic arrays with the same shape.
    necessary_mask : array-like or None
        Boolean mask selecting the non-redundant values to compare. If
        ``None``, all values are included.
    atol : float
        Maximum allowed absolute difference for each selected entry.
    verbose : bool
        Whether to print the comparison summary.

    Returns
    -------
    result : dict
        Maximum absolute and relative differences, ``atol``, and ``status``
        (``"passed"`` or ``"failed"``). Only absolute differences determine the
        status. Empty selections have zero error and pass.
    """
    matlab_values = np.asarray(matlab_array)
    plenoptic_values = np.asarray(plenoptic_array)
    assert matlab_values.shape == plenoptic_values.shape, (
        f"{name}: MATLAB shape {matlab_values.shape} != "
        f"plenoptic shape {plenoptic_values.shape}"
    )

    if necessary_mask is not None:
        matlab_values = matlab_values[..., necessary_mask]
        plenoptic_values = plenoptic_values[..., necessary_mask]

    absolute_difference = np.abs(matlab_values - plenoptic_values)
    if absolute_difference.size > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_difference = absolute_difference / np.abs(matlab_values)
        relative_difference = np.where(absolute_difference == 0, 0, relative_difference)
        max_absolute_difference = absolute_difference.max()
        mean_absolute_difference = absolute_difference.mean()
        max_relative_difference = relative_difference.max()
    else:
        # If both are empty arrays, set difference to 0
        max_absolute_difference = mean_absolute_difference = max_relative_difference = 0

    result = {
        "max_absolute_difference": max_absolute_difference,
        "max_relative_difference": max_relative_difference,
        "atol": atol,
        "status": "passed" if (absolute_difference <= atol).all() else "failed",
    }
    if verbose:
        print(f"\n{name}: {result['status']}")
        print(f"  compared values: {matlab_values.size}")
        print(f"  maximum absolute difference: {max_absolute_difference:.6g}")
        print(f"  mean absolute difference:    {mean_absolute_difference:.6g}")
        print(f"  maximum relative difference: {max_relative_difference:.6g}")
    return result


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
def compare_color_ps_statistics(matlab, atol=1e-4, verbose=True):
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
    atol : float
        Maximum allowed absolute difference for each non-redundant statistic.
    verbose : bool
        Whether to print input-image and PCA-image diagnostics and per-family
        error summaries. Diagnostics are not included in the returned rows.

    Returns
    -------
    difference_summaries : list of dict
        One row per family in ``statistics_to_compare``, in the same order.
        Each row contains ``matlab_name``, ``plenoptic_name``, the Appendix B.2
        reference ``vacher_statistic``, and ``max_absolute_difference`` and
        ``max_relative_difference`` over the selected entries, plus ``atol`` and
        ``status`` (``"passed"`` or ``"failed"``). Empty families have zero error
        and pass. Relative errors are diagnostic only. No files are written.
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
        atol=atol,
        verbose=verbose,
    )
    matlab_pca_image = torch.tensor(matlab["imPCA"], dtype=torch.float64).permute(
        2, 0, 1
    )[None]
    compare_statistic_arrays(
        "PCA-transformed image",
        matlab_pca_image,
        plenoptic_statistics["pca"]["transformed_image"],
        atol=atol,
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
        result = compare_statistic_arrays(
            label,
            matlab_representation[plenoptic_name],
            plenoptic_representation[plenoptic_name],
            plenoptic_necessary_stats[plenoptic_name],
            atol=atol,
            verbose=verbose,
        )
        difference_summaries.append(
            {
                "matlab_name": matlab_name,
                "plenoptic_name": plenoptic_name,
                "vacher_statistic": vacher_statistic,
                **result,
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
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    rows = compare_color_ps_statistics(
        load_matlab_statistics(args.matlab_path), atol=args.atol
    )
    output = args.output or Path("results") / f"{args.matlab_path.stem}_differences.csv"
    write_results(rows, output)
    print(f"\nSaved difference summaries to {output}")
    n_failed = sum(row["status"] == "failed" for row in rows)
    print(
        f"{len(rows) - n_failed}/{len(rows)} statistic families passed "
        f"(atol={args.atol})"
    )
    raise SystemExit(int(n_failed > 0))
