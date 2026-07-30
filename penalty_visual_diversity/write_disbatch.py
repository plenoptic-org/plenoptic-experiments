#!/usr/bin/env python3

import sys
import numpy as np
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/penalty_visual_diversity/")

seeds = [0]
models = ['LGC']

device = "0"
if len(sys.argv) > 2:
    fn = sys.argv[1]
    device = sys.argv[2]
elif len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = ["set -euxo pipefail"]
prefix = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"

penalty = ["pyiqa-dists_exp", "spyr-expmaskvert", "pyiqa-lpips_exp"]
# comb_func = ["exp"] + ["".join(p) for p in itertools.product(["exp", ""], ["maskall", "masklow", "maskhigh", "maskvert", "maskdiag"])]
# penalty = ['-'.join(p) for p in itertools.product(penalty, comb_func)]
# penalty = ["none", "nlpd-sse", "spyr-expmaskall", "spyr-expmaskvert", "spyr-expmaskhigh"]
imgs = ["einstein-blur1"]
for img, m, p, sd in itertools.product(imgs, models, penalty, seeds):
    if m == "PS":
        it = 1000
        # img = "reptile_skin"
    elif m == "LGC":
        it = 6000
        # img = "einstein"

    if p != "none":
        penalty_lambda = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    else:
        penalty_lambda = [1]

    for l in penalty_lambda:
        outfile = base_out / f"model-{m}_img-{img}_penalty-{p}_lambda-{l:.00e}_{device}_seed-{sd}_two-stage_iter-{it}.pt"
        if outfile.exists() or outfile.with_name(outfile.name.replace("cpu", "0")).exists():
            continue
        cmd = f"uv run --with pyiqa --with plenoptic --with pandas python synthesize.py --two-stage -m {m} -i {img} -p {p} -l {l} -d {device} -s {sd} -n {it} -f {outfile}"
        cmd = f"({prefix} {cmd}) &> {outfile.with_suffix('.log')}"
        commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
