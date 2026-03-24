#!/usr/bin/env python3

import sys
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/penalty_visual_diversity/")

seeds = [0]
models = ['LGC', 'PS']

device = "cpu"
if len(sys.argv) > 2:
    fn = sys.argv[1]
    device = sys.argv[2]
elif len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = []
prefix = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"

penalty = ["alexnet", "mse", "nlpd", "ssim", "ms_ssim"]
comb_func = ["sse", "logsumexp"]
penalty = ['-'.join(p) for p in itertools.product(penalty, comb_func)]
penalty += ["none"]
penalty_lambda = [.1, 1, 10]
for m, p, l, sd in itertools.product(models, penalty, penalty_lambda, seeds):
    if m == "PS":
        it = 200
        img = "reptile_skin"
    elif m == "LGC":
        it = "1300"
        img = "einstein"
    outfile = f"model-{m}_img-{img}_penalty-{p}_lambda-{l}_{device}_seed-{sd}_iter-{it}"
    cmd = f"python synthesize.py -m {m} -i {img} -p {p} -l {l} -d {device} -s {sd} -n {it} -f {base_out / outfile}.pt"
    cmd = f"({prefix} {cmd}) &> {base_out / outfile}.log"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
