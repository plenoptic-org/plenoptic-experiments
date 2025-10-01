# PS Speed

Investigations into comparing the performance / speed tradeoff for synthesis of Portilla-Simoncelli texture metamers using matlab and plenoptic. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details.

The results directory is created by `summarize.py` and `all_loss.csv`, summarizing that particular experiment, and some plots used to understand it.

In addition the standard scripts found in many of these experiments, this experiment uses:
- `generate_init_imgs.py`: script to generate uniform noise images to initialize synthesis (across matlab/plenoptic and optimizers).
- `example_met.py`: script giving an example of how to generate plenoptic Portilla-Simoncelli metamers with the best arguments found as part of this experiment.
