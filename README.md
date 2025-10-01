# Plenoptic experiments

Miscellaneous scripts to better understand plenoptic, to help the development of the library.

These scripts may be helpful to others, but are primarily intended for internal use. As such, they've been developed for use on the Flatiron Institute's SLURM cluster, and make use of Flatiron's [disBatch](https://github.com/flatironinstitute/disBatch) tool to submit many jobs at once. Similarly output paths are hard-coded in some of the scripts.

They are also written at a single point of time and not maintained, so they are not future-proof. I have attempted to include the specific commits when they are run.

## Some Notes About Timing

Several of these scripts are used to time / benchmark plenoptic code.

- In general, I've found [pyspy](https://github.com/benfred/py-spy), a sampling profiler, to be the best way of understanding "where is my code spending its time".
- When comparing performance, **make sure to use the same hardware for each run**. There's a roughly 2x performance difference between Flatiron's rome nodes and my workstation, with the genoa nodes having close-to-workstation performance.
    - Thus, I recommend using the genoa nodes for benchmarking cpu-only code and the A100 GPUs for benchmarking GPU code.
- Probably obvious, but you'd be surprised how often I forget: don't run something irrelevant on the same machine where you're benchmarking code.
- When running things in parallel, be careful about threading. In practice, setting the environmental variable `OMP_NUM_THREADS=1` (for OpenMP) and setting `torch.set_num_threads(1)` (or some other relatively small number) will improve performance, especially when using LBFGS.

The disbatch scripts created for each experiment are typically run like:

- cpu-only:
  ```sh
  sbatch -p ccn -C genoa -t 1-00:00:00 -n 50 -c 4 disBatch disbatch.txt
  ```

- gpu:
  ```sh
  sbatch -p gpu -C a100 --gres gpu:1 -t 1-00:00:00 -n 4 -c 4 disBatch disbatch.txt
  ```

`-c` specifies the number of cores per task, `-n` gives the number of tasks, and `--gres gpu:N` gives the number of GPUs per task. GPUs are much more in demand, so we use fewer at a time.

## Structure

Experiments live in separate directories and are independent, but often share several common elements:
- `synthesize.py` script: accepts command-line arguments and runs some synthesize.
- `time_synthesize.sh` script: calls `synthesize.py`, used so we can time how long the total script takes (include overhead of starting python and importing libraries).
- `write_disbatch.py` script: contains a giant for loop that runs through different combinations of arguments to pass to the `synthesize.py` script and writes them all out, one per line, to a txt file, which will be submitted using disbatch.
- `analyze.py` script: run after `synthesize.py`, either in parallel (in which case it will be accompanied by a `analyze_write_disbatch.py` script) or in sequence, depending on how long it takes. Loads in the outputs of `synthesize.py` and performs additional analyses on it (e.g., saving out the different computations of the loss as a `.csv`, plotting a summary figure). Should probably only be used if we are timing the duration of `synthesize.py`, otherwise its functionality should be included in `synthesize.py`
- `plot.py`: run after `analyze.py` or `synthesize.py` in sequence or parallel (in which case it will be accompanied by `plot_write_disbatch.py`). Loads in the outputs of a single `synthesize.py` / `analyze.py` run to create figures showing the output of a single run (e.g., the synthesized metamer).
- `summarize.py`: run after `analyze.py` (if it exists) or `synthesize.py` in sequence. Load in the outputs of `synthesize.py` / `analyze.py` and create a giant summary dataframe, which we plot in different ways using seaborn.

## Contents

- `ps_lbfgs/`: scripts related to understanding how to most effectively / efficiently use the LBFGS optimizer for `PortillaSimoncelli` metamer synthesis. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details. Used plenoptic commit [`aeac5144f85f0bbb1785149ea809b4ed0f7777a2`](https://github.com/plenoptic-org/plenoptic/tree/aeac5144f85f0bbb1785149ea809b4ed0f7777a2).
- `ps_matlab/`: scripts to create Portilla-Simoncelli texture metamers using the [original matlab code](https://github.com/LabForComputationalVision/textureSynth) and visualize them using plenoptic. Used plenoptic commit [`aeac5144f85f0bbb1785149ea809b4ed0f7777a2`](https://github.com/plenoptic-org/plenoptic/tree/aeac5144f85f0bbb1785149ea809b4ed0f7777a2).
- `ps_matlab_vs_plen/`: scripts for running the matlab and plenoptic Portilla-Simoncelli texture metamer synthesis, comparing the quality as a function of synthesis time. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details. Used plenoptic commit [`be60c06c9ef68c70fc363c0fef89e610e7e415ac`](https://github.com/plenoptic-org/plenoptic/tree/be60c06c9ef68c70fc363c0fef89e610e7e415ac) (importantly, involving speed-ups from the `ps_speed` branch).
- `ps_noise/`: scripts for running plenoptic's `PortillaSimoncelli` metamer synthesis with initial images that have a very different range. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details. Used plenoptic commit [`be60c06c9ef68c70fc363c0fef89e610e7e415ac`](https://github.com/plenoptic-org/plenoptic/tree/be60c06c9ef68c70fc363c0fef89e610e7e415ac) (importantly, involving speed-ups from the `ps_speed` branch).
