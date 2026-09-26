#!/usr/bin/env python3

from pathlib import Path
import itertools
import json
from collections import OrderedDict


scaling = 0.8
image = "reptile"
opt_kwargs = OrderedDict(
    line_search_fn=["strong_wolfe", None],
    lr=[0.2, 1, 3, 10],
    max_iter=[3, 10, 30, 100],
    history_size=[3, 10, 100, 300],
)

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/fenestration/")

device = "cpu"

# commands = ["set -euxo pipefail"]
commands = []
prefix = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"

keys = opt_kwargs.keys()
path_template = "_".join([f"{k}-{{{k}}}" for k in keys])
path_template = f"metamer_image-{image}_scaling-{scaling}_{path_template}.png"

for values in itertools.product(*opt_kwargs.values()):
    opt_dict = dict([(k, v) for k, v in zip(keys, values)])
    outfile = base_out / path_template.format(**opt_dict)
    opt_dict = json.dumps(opt_dict).replace('null', "None")
    cmd = f"python metamer_synth.py --device {device} --image-name {image} --scaling {scaling} --opt-kwargs '{opt_dict}' --fig-save-path {outfile}"
    commands.append(cmd)

with open("disbatch.txt", "w") as f:
    f.write("\n".join(commands))
