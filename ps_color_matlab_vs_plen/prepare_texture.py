from pathlib import Path

import imageio.v3 as iio
import torch.nn.functional as F

import plenoptic as po

# Texture to prepare and directory to save the processed image
image_name = "DSCF4315"
image_path = Path(f"{image_name}.tif")
output_path = Path(f"inputs/{image_name}.tif")

# Load the image in the 0--1 range expected by plenoptic
image = po.load_images(image_path, as_gray=False)

# Average each 4 x 4 source block, producing a deterministic antialiased
# downsample from 1024 x 1024 to 256 x 256.
image = F.interpolate(image, size=(256, 256), mode="area")

# Use the representative central portion of the bark texture.
crop_start = (256 - 128) // 2
image = image[
    ...,
    crop_start : crop_start + 128,
    crop_start : crop_start + 128,
]

# Both MATLAB and plenoptic will consume this exact 8-bit array.
image_array = po.to_numpy(image.squeeze(0).permute(1, 2, 0))
image_array = po.convert_float_to_int(image_array)

output_path.parent.mkdir(exist_ok=True)
iio.imwrite(output_path, image_array)
print(
    f"Saved {output_path} with shape {image_array.shape} and dtype {image_array.dtype}"
)
