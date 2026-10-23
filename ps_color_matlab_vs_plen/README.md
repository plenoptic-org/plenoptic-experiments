# Validation of the Portilla-Simoncelli color model

In PR #482 we are adding the color version of the Portilla-Simoncelli texture model.

The new implementation needs validating. Specifically, we want to verify that
the statistics in the plenoptic implementation are the same as in Matlab, and
that the metamers synthesized with this new model are as good as the Matlab ones.
We are tracking this in Issue #484.

The code in this directory performs some of this validation. Specifically,
it does two things:
- It computes the statistics of a texture with both Matlab and plenoptic, and
  compares that they are identical up to numerical precision.
- It synthesizes textures metamers with both Matlab and plenoptic.

This validation suite has the following components:
- `prepare_texture.py`: Takes a larger image and turns it into a 128 x 128 image saved in `inputs/` for the analyses to consume
- `run_matlab.m`: Computes the color statistics of the image, and generates a metamer. Statistics are saved in `matlab_statistics/`, and the metamer in `matlab_metamers/`. Also saves the PCA parameters for exact reproducibility in Python.
- `compute_stats_plenoptic.py`: Computes the color Portilla-Simoncelli statistics with the new implementation. It loads and uses the PCA parameters obtained by Matlab, for full reproducibility. Saves statistics in `plenoptic_statistics/`.
- `synthesize_plenoptic.py`: Synthesize metamers with plenoptic, using the PCA and OPC color transforms and both the default and color-matched initializations. It doesn't do any Matlab matching. Saves the metamers in `plenoptic_metamers/`.
- `color_test_helpers.py`: Extends plenoptic's (pre PR 482) test helpers with shared grayscale/color normalization functions and a MATLAB color dictionary mapper.
- `compare_statistics.py`: Loads the saved statistics, calls the conversion helpers, and prints the absolute and relative differences for each statistic family. Saves the summaries to `results/matlab_plen_differences.csv`.

Run the analysis from `ps_color_matlab_vs_plen/`. First set `image_path` in `prepare_texture.py` to an available image. Adjust the image names in the other scripts as well. Then run, in order, `prepare_texture.py`, `run_matlab.m`, `synthesize_plenoptic.py`, `compute_stats_plenoptic.py`, and finally `compare_statistics.py`.

## Converting MATLAB statistics

`color_test_helpers.py` contains local versions of `construct_normalizing_dict`
and `remove_redundant_and_normalize`, based directly on plenoptic's
`tests/test_models.py`. Each function handles both grayscale and color through
`plen_ps.color_statistics`. The original calculation order is retained, with
conditional rearrangements before the shared `einsum` operations and additional
lines for color-only statistics.

`matlab_to_plenoptic_dict_color(params, n_scales, n_orientations)` renames MATLAB
fields, rearranges axes, and selects active blocks. Both grayscale and color use
the two-argument call
`construct_normalizing_dict(model, image)`.
Both modes use `remove_redundant_and_normalize(rep, False, model, norm_dict)` to
normalize raw covariances and pack the representation with the model's masks.

All cross-correlation normalizers use variances computed from the image's pyramid
intermediates. Supply the already transformed MATLAB PCA image when constructing
the color normalizers. For lowpass normalizers, the expanded residual is divided
by four to match MATLAB's covariance convention; periodic shifts preserve its
variance. The helpers compute all joint channel/orientation factors directly,
without an adapter model or per-channel grayscale conversion.

## Getting the MATLAB version of the code

Before running the analysis, download the Matlab dependencies into
`plenoptic-experiments/ps_color_matlab_vs_plen/matlab_ps_code/`:

- [matlabPyrTools](https://github.com/LabForComputationalVision/matlabPyrTools): the pyramid tools.
- [textureSynth](https://github.com/LabForComputationalVision/textureSynth): the grayscale texture model and shared functions.
- [colorTextureSynth](https://www.cns.nyu.edu/pub/eero/colorTextureSynth.tgz): the color texture model.

One option is to run the following from the `plenoptic-experiments` repository root:

```sh
cd ps_color_matlab_vs_plen
mkdir -p matlab_ps_code
git clone https://github.com/LabForComputationalVision/matlabPyrTools.git matlab_ps_code/matlabPyrTools
git clone https://github.com/LabForComputationalVision/textureSynth.git matlab_ps_code/textureSynth
curl -fL https://www.cns.nyu.edu/pub/eero/colorTextureSynth.tgz -o matlab_ps_code/colorTextureSynth.tgz
tar -xzf matlab_ps_code/colorTextureSynth.tgz -C matlab_ps_code
```

This should download the three dependencies into `plenoptic-experiments/ps_color_matlab_vs_plen/matlab_ps_code`. The MATLAB script `run_matlab.m` adds `matlab_ps_code/` and its subdirectories to the Matlab path automatically.
