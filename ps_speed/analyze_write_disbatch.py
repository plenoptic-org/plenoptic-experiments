#!/usr/bin/env python3

import sys
import pathlib

base_dir = pathlib.Path(sys.argv[1])

commands = []
for f in base_dir.glob("*mat"):
    if not f.with_suffix(".csv").exists():
        cmd = f"PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic python analyze.py {f}"
        commands.append(cmd)

for f in base_dir.glob("*pt"):
    if not f.with_suffix(".csv").exists():
        cmd = f"PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic python analyze.py {f}"
        commands.append(cmd)

with open("disbatch_analyze.txt", "w") as f:
    f.write("\n".join(commands))
