# Plenoptic experiments

Miscellaneous scripts to better understand plenoptic, to help the development of the library.

These scripts may be helpful to others, but are primarily intended for internal use. As such, they've been developed for use on the Flatiron Institute's SLURM cluster, and make use of Flatiron's [disBatch](https://github.com/flatironinstitute/disBatch) tool to submit many jobs at once.

They are also written at a single point of time and not maintained, so they are not future-proof. I have attempted to include the specific commits when they are run.

## Structure

Experiments live in separate directories and are independent, but often share several common elements:
- `synthesize.py` script: accepts command-line arguments and runs some synthesize.
- `write_disbatch.py` script: contains a giant for loop that runs through different combinations of arguments to pass to the `synthesize.py` script and writes them all out, one per line, to a txt file, which will be submitted using disbatch.
- `analyze.py` script: run after `synthesize.py`, either in parallel (in which case it will be accompanied by a `write_disbatch_analyze.py` script) or in sequence, depending on how long it takes. Loads in the outputs of `synthesize.py` and performs additional analyses on it (e.g., saving out the different computations of the loss as a `.csv`, plotting a summary figure). Should probably only be used if we are timing the duration of `synthesize.py`, otherwise its functionality should be included in `synthesize.py`
- `summarize.py`: run after `analyze.py` (if it exists) or `synthesize.py` in sequence. Load in the outputs of `synthesize.py` / `analyze.py` and create a giant summary dataframe, which we plot in different ways using seaborn.

## Contents

- `ps_lbfgs/`: scripts related to understanding how to most effectively / efficiently use the LBFGS optimizer for `PortillaSimoncelli` metamer synthesis. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details.
- `ps_matlab/`: scripts to create Portilla-Simoncelli texture metamers using the [original matlab code](https://github.com/LabForComputationalVision/textureSynth) and visualize them using plenoptic.
- `ps_speed/`: scripts for running the matlab and plenoptic Portilla-Simoncelli texture metamer synthesis, comparing the quality as a function of synthesis time. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details.
- `ps_noise/`: scripts for running plenoptic's `PortillaSimoncelli` metamer synthesis with initial images that have a very different range. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details.
