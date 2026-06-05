# plenoptic GLM

Little repository to use plenoptic with a simple GLM from the [NeMoS GLM tutorials](https://balzaniedoardo.github.io/nemos_glm_tutorials/index.html).

To run:

```sh
uv sync --extras nemos
# use nemos to fit data and save results
uv run python nemos_glm.py
# validate the plenoptic and nemos glms are the same
uv run python plenoptic_glm.py validate-model glm_stim.npz stim
uv run python plenoptic_glm.py validate-model glm_stim_spk.npz stim_spk
# synthesize and plot metamers, no penalty
uv run python plenoptic_glm.py synthesize glm_stim.npz nemos_stimulus.npz --save-stem glm_met --seed 0
uv run python plenoptic_glm.py plot glm_stim.npz nemos_stimulus.npz --load-stem glm_met --metamer-label "Metamer"
# synthesize and plot metamers, range penalty
uv run python plenoptic_glm.py synthesize glm_stim.npz nemos_stimulus.npz --penalty range --save-stem glm_met-range --seed 0
uv run python plenoptic_glm.py plot glm_stim.npz nemos_stimulus.npz --penalty range --load-stem glm_met-range --metamer-label "Metamer (Range)"
# synthesize and plot metamers, MSE penalty
uv run python plenoptic_glm.py synthesize glm_stim.npz nemos_stimulus.npz --penalty mse --save-stem glm_met-mse --seed 1 --penalty-lambda 0.001
uv run python plenoptic_glm.py plot glm_stim.npz nemos_stimulus.npz --penalty mse --load-stem glm_met-mse --metamer-label "Metamer (Range+MSE)" --penalty-lambda 0.001

# and all the same for the glm_stim_spk:
uv run python plenoptic_glm.py synthesize glm_stim_spk.npz nemos_stimulus.npz --save-stem glm_spk_met --seed 0
uv run python plenoptic_glm.py plot glm_stim_spk.npz nemos_stimulus.npz --load-stem glm_spk_met --metamer-label "Metamer"
uv run python plenoptic_glm.py synthesize glm_stim_spk.npz nemos_stimulus.npz --penalty range --save-stem glm_spk_met-range --seed 0
uv run python plenoptic_glm.py plot glm_stim_spk.npz nemos_stimulus.npz --penalty range --load-stem glm_spk_met-range --metamer-label "Metamer (Range)"
uv run python plenoptic_glm.py synthesize glm_stim_spk.npz nemos_stimulus.npz --penalty mse --save-stem glm_spk_met-mse --seed 1 --penalty-lambda 0.001
uv run python plenoptic_glm.py plot glm_stim_spk.npz nemos_stimulus.npz --penalty mse --load-stem glm_spk_met-mse --metamer-label "Metamer (Range+MSE)" --penalty-lambda 0.001
```
