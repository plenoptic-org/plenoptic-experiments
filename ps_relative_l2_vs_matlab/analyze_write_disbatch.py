#!/usr/bin/env python3

import sys
import pathlib

base_dir = pathlib.Path(sys.argv[1])

if len(sys.argv) > 2:
    fn = sys.argv[2]
else:
    fn = "disbatch_analyze.txt"

commands = []
for f in base_dir.glob("*mat"):
    if not f.with_suffix(".csv").exists():
        cmd = f"PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic python analyze.py {f}"
        commands.append(cmd)

for f in base_dir.glob("*pt"):
    if not f.with_suffix(".csv").exists():
        cmd = f"PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic python analyze.py {f}"
        commands.append(cmd)

with open(fn, "w") as f:
    f.write("\n".join(commands))
