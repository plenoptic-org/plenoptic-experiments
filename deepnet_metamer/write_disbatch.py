#!/usr/bin/env python3

import sys
import numpy as np
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/deepnet_metamer/")

seeds = [0, 1]
it = 12000

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

scheduler = [None, "StepLR-3000", "StepLR-1000"]
optimizer = ["Adam"]
loss_func = ["mse", "l2_norm"]
lr = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
for s, sch, o, loss, l in itertools.product(seeds, scheduler, optimizer, loss_func, lr):
    outfile = base_out / f"model-ResNet50_img-parrot_{device}_seed-{s}_iter-{it}_loss-{loss}_opt-{o}_lr-{l:.1e}_sch-{sch}.pt"
    if outfile.exists():
        continue
    cmd = f"python synthesize.py -d {device} -s {s} -n {it} -o {o} -l {l:.1e} -c {sch} --loss {loss} -f {outfile}"
    cmd = f"({prefix} {cmd}) &> {outfile.with_suffix('.log')}"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
