#!/usr/bin/env python3

import sys
import numpy as np
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/penalty_visual_diversity_2/")

seeds = [0]
models = ['LGC']

device = "cpu"
if len(sys.argv) > 2:
    fn = sys.argv[1]
    device = sys.argv[2]
elif len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

# commands = ["set -euxo pipefail"]
commands = []
prefix = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"

imgs = ["einstein-blur1", "einstein"]
max_iter = 1000
penalty_lambda = {
    "mse": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "l2_norm": [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
}
lr = [0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.1]
loss_func = ["l2_norm", "mse", "l2_norm-mse"]
for img, m, l, f, sd in itertools.product(imgs, models, lr, loss_func, seeds):
    for p in penalty_lambda[f.split("-")[-1]]:
        outfile = base_out / f"model-{m}_img-{img}_lr-{l:.00e}_loss-{f}_penalty-dists_lambda-{p:.00e}_{device}_seed-{sd}_two-stage_iter-{max_iter}.pt"
        if outfile.exists() or outfile.with_name(outfile.name.replace("cpu", "0")).exists():
            continue
        cmd = f"python synthesize.py --two-stage -m {m} -i {img} -p dists -l {p} --lr {l} --loss_func {f} -d {device} -s {sd} -n {max_iter} -f {outfile}"
        cmd = f"({prefix} {cmd}) &> {outfile.with_suffix('.log')}"
        commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
