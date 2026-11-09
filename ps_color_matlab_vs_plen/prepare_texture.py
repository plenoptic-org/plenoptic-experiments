"""Prepare identical 8-bit RGB texture inputs for MATLAB and plenoptic."""

import argparse
from pathlib import Path

import imageio.v3 as iio

import plenoptic as po

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("images", nargs="+", type=Path)
parser.add_argument("--output-dir", type=Path, default=Path("inputs"))
parser.add_argument("--crop-size", type=int, default=256)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

for image_path in args.images:
    image = po.load_images(image_path, as_gray=False)
    image = po.process.center_crop(image, args.crop_size)
    image_array = po.to_numpy(image.squeeze(0).permute(1, 2, 0))
    image_array = po.convert_float_to_int(image_array)
    output_path = args.output_dir / f"{image_path.stem}.tif"
    iio.imwrite(output_path, image_array)
    print(
        f"Saved {output_path} with shape {image_array.shape} "
        f"and dtype {image_array.dtype}"
    )
