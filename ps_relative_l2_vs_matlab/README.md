# PS relative L2 vs matlab

Investigations into comparing the performance / speed tradeoff for synthesis of Portilla-Simoncelli texture metamers using matlab and plenoptic, using the relative L2 loss. See [issue #365](https://github.com/plenoptic-org/plenoptic/issues/365) for more details.

The results directory is created by `summarize.py` and contains `all_loss.csv`, summarizing that particular experiment, and some plots used to understand it.

In addition the standard scripts found in many of these experiments, this experiment uses:
- `generate_init_imgs.py`: script to generate uniform noise images to initialize synthesis (across matlab/plenoptic and optimizers).
