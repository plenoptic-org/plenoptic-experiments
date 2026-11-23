# Loss changes for color Portilla-Simoncelli synthesis

With the color Portilla-Simoncelli model of PR #482, metamers look good when
synthesis starts from noise matched to the target's RGB mean and covariance.
Starting from plain uniform noise, they look washed out. Here we test whether
changes to the loss fix this without the color-matched initialization.

Run the scripts from `ps_color_loss/`.

## Previous findings

An earlier exploratory run found that the washed-out look comes from the
RGB **mean**, which stays close to the gray of the uniform noise, while
both initializations match the target RGB covariance within about 1.4%.

## Inputs from `ps_color_matlab_vs_plen`

This experiment reuses the images, MATLAB metamers and code of
[`ps_color_matlab_vs_plen`](../ps_color_matlab_vs_plen/README.md). Prepare the
input images and synthesize the MATLAB metamers there first:

- `../ps_color_matlab_vs_plen/sweep.json`: the list of images.
- `../ps_color_matlab_vs_plen/inputs/`: the prepared 256 x 256 textures.
- `../ps_color_matlab_vs_plen/matlab_metamers/`: the MATLAB metamers.

## Synthesizing metamers

`synthesize.py` synthesizes a metamer for each image with the PCA transform and
the default loss, starting from uniform noise (`uniform`) and from color-matched
noise (`color_matched`). It saves them in `metamers/`.

```sh
python synthesize.py
```
