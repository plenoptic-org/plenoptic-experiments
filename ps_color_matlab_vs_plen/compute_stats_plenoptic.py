"""Compute color PS statistics using plenoptic, starting from the image and
PCA transform exported by MATLAB."""

import torch

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
