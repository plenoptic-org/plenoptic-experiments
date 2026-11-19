# Validation of the Portilla-Simoncelli color model

In PR #482 we are adding the color version of the Portilla-Simoncelli texture model.

The new implementation needs validating. Specifically, we want to verify that
the statistics in the plenoptic implementation are the same as in Matlab, and
that the metamers synthesized with this new model are as good as the Matlab ones.
We are tracking this in Issue #484.

Run the commands below from `ps_color_matlab_vs_plen/`.

The components needed to **compare the statistics** of MATLAB and plenoptic are the following:

- `export_matlab_ps_stats.m` generates one MATLAB reference for each case in `sweep.json`.
- `compare_sweep.py` compares those references locally and writes per-family and per-case reports.
- `matlab_statistics_helpers.py` maps MATLAB statistics to plenoptic's layout and provides shared grayscale and color normalization functions.
- `compare_statistics.py` compares a single MATLAB reference with plenoptic statistics computed using the reference PCA parameters. It reports absolute and relative differences for each statistic family and saves a CSV in `results/`.

The components needed to **synthesize metamers** are the following:

- `export_matlab_ps_metamer.m` synthesizes a metamer for one image using baseline MATLAB parameters and saves the metamer and synthesis metadata in `matlab_metamers/`.
- `synthesize_plenoptic.py` synthesizes metamers using the PCA and OPC color transforms with both default and color-matched initializations. It saves the outputs in `plenoptic_metamers/`.

## Preparing input images

Run this once to save a 256 x 256 center crop of each source texture in `inputs/`
for both statistics comparison and metamer synthesis.

```sh
python prepare_texture.py xzhao_textures/*.tif
```

## Comparing multiple images and parameters

`sweep.json` defines parameter combinations for each of the four textures.
To change the sweep, edit the lists in that file.

**Generate MATLAB references on the cluster.** Use the same `sweep.json` and
[prepared images](#preparing-input-images) on both machines. With the MATLAB
dependencies installed, run the exporter from the experiment directory.

```matlab
export_matlab_ps_stats
```

**Compare locally.** Copy the generated `validation/matlab_statistics/` directory
back to the same relative path and run the comparison.

```sh
python compare_sweep.py
```

Check `validation/case_summary.csv` for pass/fail by case and
`validation/comparisons.csv` for errors by statistic family.

## Comparing plenoptic to MATLAB outputs

To run a comparison, the matlab output files need to be generated first,
because plenoptic computes its statistics using matlab outputs for the sake
of comparison. Once the .mat file is generated, compare a MATLAB reference with
a fresh plenoptic computation:

```sh
python compare_statistics.py matlab_statistics/matlab_d30_2180.mat
```

This writes `results/matlab_d30_2180_differences.csv`. Use `--output` to choose
another output file. When no reference path is supplied, the command defaults to
`matlab_statistics/matlab_DSCF4315.mat`.

A statistic family passes when every non-redundant entry satisfies
`abs(plenoptic - matlab) <= atol`. Statistics families that are empty in both
implementations pass (e.g. cross-scale correlations when there's a
single scale). The CSV includes `atol` and per-family `status` columns. The
command writes the report and exits with status 1 if any family fails, or 0
if all pass.

## Generating a baseline MATLAB metamer

Run `export_matlab_ps_metamer` in MATLAB on the cluster. By default it
synthesizes a metamer for `inputs/DSCF4315.tif` and saves the image and
synthesis metadata in `matlab_metamers/`.

## Loading and converting MATLAB statistics

`matlab_statistics_helpers.py` contains local versions of the helper functions of plenoptic's `tests/test_models.py`.
Each local helper function handles both grayscale and color.

`load_matlab_statistics(path)` reads MATLAB reference files without messing up the dimensions in the process, and unpacks the `params` struct into a dictionary.

`matlab_to_plenoptic_dict_color(params, n_scales, n_orientations)` renames MATLAB fields, rearranges axes, and selects active blocks. Both grayscale and color use `remove_redundant_and_normalize(rep, False, model, norm_dict)` to normalize raw covariances and pack the representation with the model's masks.

Cross-correlation normalizers are computed from the image's pyramid intermediates. The helpers compute all joint channel/orientation factors directly, without an adapter model or per-channel grayscale conversion.

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

This should download the three dependencies into `plenoptic-experiments/ps_color_matlab_vs_plen/matlab_ps_code`. Both MATLAB exporters add `matlab_ps_code/` and its subdirectories to the Matlab path by default.
