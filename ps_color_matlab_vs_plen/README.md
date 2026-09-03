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
- `synthesize_plenoptic.py`: Synthesize a metamer with plenoptic, using the PCA and the OPC color transforms. It doesn't do any Matlab matching. Saves the metamer in `plenoptic_metamers/`.
- `compare_statistics.py`: Compares the Matlab and the plenoptic color Portilla-Simoncelli statistics, going family by family. Essentially, it converts Matlab statistics, that are covariances, into correlations like the ones computed by `plenoptic`. The script just prints the relative and absolute differences between the two implementations statistics.

To run the analysis, first set `image_path` in `prepare_texture.py` to an available image. Adjust the image names in the other scripts as well. Then run, in order, `prepare_texture.py`, `run_matlab.m`, `synthesize_plenoptic.py`, `compute_stats_plenoptic.py`, and finally `compare_statistics.py`.
