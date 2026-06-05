"""Basic re-implementation of model from nemos tutorial.

Fit the model, save the parameters, save input/output on some random data (to verify
behavior).

Working from MAP Decoding notebook
https://balzaniedoardo.github.io/nemos_glm_tutorials/tutorials/Sfn-2016-tutorial-GLMs/05_decoding.html

"""

import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import pynapple as nap
import nemos as nmo
from nemos_tutorials import fetch_data, PALETTE

## Load and pre-process data

data_paths = fetch_data("data_RGCs")

# Load and wrap spike times
spike_times = loadmat(data_paths["SpTimes.mat"], simplify_cells=True)["SpTimes"]
units = nap.TsGroup({i: nap.Ts(val) for i, val in enumerate(spike_times)})

# Load and wrap stimulus
stim_times = loadmat(data_paths["stimtimes.mat"], simplify_cells=True)["stimtimes"]
stim = loadmat(data_paths["Stim.mat"], simplify_cells=True)["Stim"]
stimulus = nap.Tsd(stim_times, stim)

# Align, count, resample
units = units.restrict(stimulus.time_support)
bin_size = stimulus.t[1] - stimulus.t[0]
counts = units.count(bin_size, stimulus.time_support)
stimulus = counts.value_from(stimulus, mode="before")

cell_idx = 2
neuron_counts = counts[:, cell_idx]

## Train/test split

n_train = int(stimulus.size * (1 / 4))
n_test = 50
window_size_stim = 20

# The test window starts after the training data, offset by the filter length
# so that the design matrix has no NaN-padded rows in the test window.
n_test_start = n_train + window_size_stim - 1
n_test_stop  = n_test_start + n_test

y_test = neuron_counts[n_test_start:n_test_stop]

## Fit forward models

### Stimulus-only GLM

basis_stim = nmo.basis.HistoryConv(window_size_stim, label="stim", conv_kwargs={"shift": False})
X_stim = basis_stim.compute_features(stimulus[:n_train])
y_train = neuron_counts[:n_train]

glm_stim = nmo.glm.GLM(observation_model="Poisson")
glm_stim.fit(X_stim, y_train)

glm_stim.save_params("glm_stim.npz")

### GLM with spike history and coupling

window_size_spk = 20

basis_spk  = nmo.basis.HistoryConv(window_size_spk,  label="spike")
basis_stim_spk = basis_stim + basis_spk

X_stim_spk = basis_stim_spk.compute_features(stimulus[:n_train], counts[:n_train])
X_stim_spk = {k: v.reshape((n_train, -1)) for k, v in  basis_stim_spk.split_by_feature(X_stim_spk).items()}
glm_stim_spk = nmo.glm.GLM(observation_model="Poisson", solver_name="BFGS")
glm_stim_spk.fit(X_stim_spk, y_train)

glm_stim_spk.save_params("glm_stim_spk.npz")

save_dict = {}
save_dict["n_simulations"] = 5
for i in range(save_dict["n_simulations"]):
    key = jax.random.PRNGKey(i)
    rand_input = jax.random.uniform(key, 10000)
    X_rand = basis_stim.compute_features(rand_input)
    output_stim = glm_stim.predict(X_rand)
    zeros_spk = basis_spk.compute_features(jax.numpy.zeros((rand_input.shape[0], counts.shape[1])))
    output_stim_spk = glm_stim_spk.predict({"stim": X_rand, "spike": zeros_spk})
    save_dict[f"input_{i}"] = rand_input
    save_dict[f"output_stim_{i}"] = output_stim
    save_dict[f"output_stim_spk_{i}"] = output_stim_spk

np.savez("nemos_simulations.npz", allow_pickle=False, **save_dict)
