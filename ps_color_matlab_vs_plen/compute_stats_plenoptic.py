from pathlib import Path

import scipy.io as sio
import torch

import plenoptic as po

DTYPE = torch.float64

# Texture to compute synthesis for. Will be taken from the matlab output
image_name = "DSCF4315"
matlab_path = Path(f"matlab_statistics/matlab_{image_name}.mat")

# Dir to save the plenoptic computed statistics
statistics_dir = Path("plenoptic_statistics")
statistics_dir.mkdir(exist_ok=True)

# Load image and parameters as used by matlab
matlab = sio.loadmat(matlab_path, simplify_cells=True)
image = torch.tensor(matlab["im0"], dtype=DTYPE).permute(2, 0, 1).unsqueeze(0)

n_scales = int(matlab["Nsc"])
n_orientations = int(matlab["Nor"])
spatial_corr_width = int(matlab["Na"])
matlab_transformed_image = (
    torch.tensor(matlab["imPCA"], dtype=DTYPE).permute(2, 0, 1).unsqueeze(0)
)

# Reproduce the matlab PCA parameters, for exact numeric comparison
# (e.g. the sign of PCA components is arbitrary)
pca_matlab = po.process.PCA(image, max_relative_scaling=float("inf"))
with torch.no_grad():
    pca_matlab.mean.copy_(torch.tensor(matlab["pcaMean"], dtype=DTYPE).reshape(3, 1, 1))
    pca_matlab.matrix.copy_(torch.tensor(matlab["pcaMatrix"], dtype=DTYPE))

model = po.models.PortillaSimoncelli(
    image.shape[-2:],
    n_scales=n_scales,
    n_orientations=n_orientations,
    spatial_corr_width=spatial_corr_width,
    color_statistics=True,
    transform=pca_matlab,
).to(dtype=DTYPE)
model.eval()
po.remove_grad(model)

with torch.no_grad():
    representation_tensor = model(image)
    representation = model.convert_to_dict(representation_tensor)
    transformed_image = pca_matlab(image)

# Check that the PCA transformed image of plenoptic is the same as the
# one generated in Matlab
pca_max_abs_difference = (transformed_image - matlab_transformed_image).abs().max()
torch.testing.assert_close(
    transformed_image,
    matlab_transformed_image,
    rtol=1e-10,
    atol=1e-10,
)

# Save the statistics
statistics = {
    "image": image,
    "pca": {
        "mean": pca_matlab.mean,
        "matrix": pca_matlab.matrix,
        "transformed_image": transformed_image,
        "matlab_transformed_image": matlab_transformed_image,
        "max_abs_difference": pca_max_abs_difference,
    },
    "model": {
        "n_scales": n_scales,
        "n_orientations": n_orientations,
        "spatial_corr_width": spatial_corr_width,
    },
    "representation_tensor": representation_tensor,
    # This is the actual OrderedDict returned by model.convert_to_dict().
    "representation": representation,
    "necessary_stats": model._necessary_stats_dict,
}

statistics_path = statistics_dir / f"plenoptic_matlab_pca_{image_name}.pt"
torch.save(statistics, statistics_path)
print(f"Saved plenoptic target statistics to {statistics_path}")
