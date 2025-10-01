import torch
import sys
import plenoptic as po
import subprocess
import pathlib
import time
import numpy as np
from plenoptic.data.fetch import fetch_data

po_dir = pathlib.Path(po.__file__).parent.parent.parent
out = subprocess.run(["git", "branch", "--show-current"], cwd=po_dir,
                     capture_output=True)
branch_name = out.stdout.decode().strip()
print(f"plenoptic location: {po_dir}")
print(f"git branch: {branch_name}")

img_dir = fetch_data("portilla_simoncelli_images.tar.gz")
torch.set_num_threads(1)

device = "cpu"
if len(sys.argv) > 1:
    device = sys.argv[1]

device = torch.device(device)
print(f"Device: {device.type}")

duration_long = 10
synth_n = 1
synth_loop_before_reset = 1
each_synth = 1
synth_kwargs = {"stop_criterion": 1e-10}
timing = {}

img = po.load_images(img_dir / "fig4a.jpg").to(torch.float64).to(device)
ps = po.simul.PortillaSimoncelli(img.shape[-2:])
ps.to(device)
pyr_no_ds = po.simul.SteerablePyramidFreq(img.shape[-2:], downsample=False, is_complex=True)
pyr_no_ds.to(device)
po.tools.set_seed(0)
init = torch.rand_like(img)

print(f"pyramid forward")
times = []
for i in range(duration_long):
    start = time.time()
    ps._pyr(img)
    times.append(time.time() - start)
timing["pyramid_forward"] = np.asarray(times)
print(f"mean: {np.mean(times)}, stdev: {np.std(times)}")

print("pyramid recon")
coeffs = ps._pyr(img)
times = []
for i in range(duration_long):
    start = time.time()
    ps._pyr.recon_pyr(coeffs)
    times.append(time.time() - start)
timing["pyramid_recon"] = np.asarray(times)
print(f"mean: {np.mean(times)}, stdev: {np.std(times)}")

coeffs = pyr_no_ds(img)
for split_complex in [True, False]:
    print(f"pyramid convert_pyr_to_tensor split_complex={split_complex}")
    times = []
    for i in range(duration_long):
        start = time.time()
        pyr_no_ds.convert_pyr_to_tensor(coeffs, split_complex=split_complex)
        times.append(time.time() - start)
    timing[f"pyramid_convert_pyr_to_tensor_split-{split_complex}"] = np.asarray(times)
    print(f"mean: {np.mean(times)}, stdev: {np.std(times)}")

    coeffs_tensor, pyr_info = pyr_no_ds.convert_pyr_to_tensor(coeffs, split_complex=split_complex)
    print(f"pyramid convert_tensor_to_pyr split_complex={split_complex}")
    times = []
    for i in range(duration_long):
        start = time.time()
        pyr_no_ds.convert_tensor_to_pyr(coeffs_tensor, *pyr_info)
        times.append(time.time() - start)
    timing[f"pyramid_convert_tensor_to_pyr_split-{split_complex}"] = np.asarray(times)
    print(f"mean: {np.mean(times)}, stdev: {np.std(times)}")

print("ps forward")
times = []
for i in range(duration_long):
    start = time.time()
    ps(img)
    times.append(time.time() - start)
timing["ps_forward"] = np.asarray(times)
print(f"mean: {np.mean(times)}, stdev: {np.std(times)}")

print("metamer Adam")

times = []
for i in range(synth_n):
    t = []
    met = po.synth.MetamerCTF(img, ps, loss_function=po.tools.optim.l2_norm)
    met.setup(init)
    for j in range(synth_loop_before_reset):
        start = time.time()
        met.synthesize(each_synth, **synth_kwargs)
        t.append(time.time() - start)
    times.append(t)
timing["metamer_adam_synth-10"] = np.asarray(times)
print(f"mean: {np.mean(times)}, stdev: {np.std(times)}")

weights = ps.convert_to_dict(torch.ones_like(ps(img)))
for k in ["var_highpass_residual"]:
    weights[k] = 100 * torch.ones_like(weights[k])
weights = ps.convert_to_tensor(weights)
def loss(x, y):
    return po.tools.optim.l2_norm(weights*x, weights*y)

for history_size in [3, 100, 300]:
    print(f"metamer LBFGS history_size={history_size}")
    times = []
    optimizer_kwargs = {"line_search_fn": "strong_wolfe", "history_size": history_size,
                        "max_iter": 10, "max_eval": 10, "lr": 1}
    for i in range(synth_n):
        t = []
        met_lbfgs = po.synth.Metamer(img, ps, loss_function=loss)
        met_lbfgs.setup(init, optimizer=torch.optim.LBFGS, optimizer_kwargs=optimizer_kwargs)
        for j in range(synth_loop_before_reset):
            start = time.time()
            met_lbfgs.synthesize(each_synth, **synth_kwargs)
            t.append(time.time() - start)
        times.append(t)
    timing[f"metamer_lbfgs_synth-10_history-{history_size}"] = np.asarray(times)
    print(f"mean: {np.mean(times)}, stdev: {np.std(times)}")

np.savez(f"ps_speed_timing_{device.type}_{branch_name}", **timing)
