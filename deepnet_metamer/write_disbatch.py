#!/usr/bin/env python3

import sys
import numpy as np
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/deepnet_metamer/")

seeds = [0, 1]
it = 12000

device = "cpu"
if len(sys.argv) > 2:
    fn = sys.argv[1]
    device = sys.argv[2]
elif len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = ["set -euxo pipefail"]
prefix = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"

scheduler = ["StepLR-5000", "StepLR-3000", "StepLR-1000", None]
optimizer = ["Adam"]
loss_func = ["mse"]
layer = ["layer4", "layer2", "layer3"]
lr = [1e-3, 3e-3, 3e-2, 1e-2, 1e-1]
plot_cmds = []
synth_cmds = []
images = ["macaque"]
for s, sch, o, loss, l, lay, img in itertools.product(seeds, scheduler, optimizer, loss_func, lr, layer, images):
    outfile = base_out / f"model-ResNet50-{lay}_img-{img}_{device}_seed-{s}_iter-{it}_loss-{loss}_opt-{o}_lr-{l:.1e}_sch-{sch}.pt"
    if outfile.exists() or outfile.with_name(outfile.name.replace("cpu", "0")).exists() or outfile.with_name(outfile.name.replace("_0_", "_cpu_")).exists():
        cmd = f"python plot.py --loss {loss} --layer {lay} --image {img} --plot -f {outfile}"
        cmd = f"{prefix} {cmd}"
        plot_cmds.append(cmd)
    else:
        cmd = f"python synthesize.py -d {device} -s {s} -n {it} -o {o} --image {img} -l {l:.1e} -c {sch} --loss {loss} --layer {lay} -f {outfile}"
        cmd = f"({prefix} {cmd}) &> {outfile.with_suffix('.log')}"
        synth_cmds.append(cmd)

# commands.extend(plot_cmds)
commands.extend(synth_cmds)
with open(fn, "w") as f:
    f.write('\n'.join(commands))
