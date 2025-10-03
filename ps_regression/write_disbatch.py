#!/usr/bin/env python3
import pathlib
import sys

old_dir = pathlib.Path(sys.argv[1])
new_dir = pathlib.Path(sys.argv[2])
out_dir = new_dir.parent / "figs"
out_dir.mkdir(exist_ok=True)

prepend = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"

commands = []
for f in old_dir.glob("ps*pt"):
    out_file = out_dir / ('old_' + f.with_suffix('.svg').name)
    cmd = f"{prepend} python plot.py {f} {out_file} &> {out_file.with_suffix('.log')}"
    commands.append(cmd)

for f in new_dir.glob("ps*pt"):
    out_file = out_dir / ('new_' + f.with_suffix('.svg').name)
    cmd = f"{prepend} python plot.py {f} {out_file} &> {out_file.with_suffix('.log')}"
    commands.append(cmd)

with open("disbatch.txt", "w") as f:
    f.write("\n".join(commands))
