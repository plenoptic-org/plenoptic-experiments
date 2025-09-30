#!/usr/bin/env python3

import sys
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_speed/matlab_results")

synth_iters = [400, 800]
seeds = list(range(10))
figs = ["fig4a", "fig16b"]

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = []
for f, it, sd in itertools.product(figs, synth_iters, seeds):
    outfile = f"{f}_seed-{sd}_iter-{it}"
    cmd = f"bash time_matlab_synthesize.sh {f} {it} seed-{sd}"
    cmd = f"({cmd}) &> {base_out / outfile}.log"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
