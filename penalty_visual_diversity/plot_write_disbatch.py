#!/usr/bin/env python3

import sys
from pathlib import Path
base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/penalty_visual_diversity/")

commands = []
prefix = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "plot_disbatch.txt"

for f in base_out.glob("*pt"):
    if f.with_suffix(".svg").exists():
        continue
    cmd = f"{prefix} python plot.py {f}"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
