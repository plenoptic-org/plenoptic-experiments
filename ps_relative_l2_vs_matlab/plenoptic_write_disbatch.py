#!/usr/bin/env python3

import sys
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_relative_l2_vs_matlab/plenoptic_results/")
base_out.mkdir(exist_ok=True)

seeds = list(range(10))
figs = ["fig4a", "fig16b"]

device = "cpu"
if len(sys.argv) > 2:
    fn = sys.argv[1]
    device = sys.argv[2]
elif len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = []

optimizer = "Adam"
synth_iters = [30, 75, 150, 300, 600]
for f, it, sd in itertools.product(figs, synth_iters, seeds):
    outfile = f"{optimizer}_{device}_{f}_seed-{sd}_iter-{it}"
    cmd = f"bash time_plenoptic_synthesize.sh {f} seed-{sd} {optimizer} 0 {device} {it} {base_out / outfile}.pt"
    cmd = f"({cmd}) &> {base_out / outfile}.log"
    commands.append(cmd)

optimizer = "LBFGS"
synth_iters = [10, 25, 50, 100, 200]
history_size = [10, 100]
for f, it, sd, h in itertools.product(figs, synth_iters, seeds, history_size):
    outfile = f"{optimizer}-{h}_{device}_{f}_seed-{sd}_iter-{it}"
    cmd = f"bash time_plenoptic_synthesize.sh {f} seed-{sd} {optimizer} {h} {device} {it} {base_out / outfile}.pt"
    cmd = f"({cmd}) &> {base_out / outfile}.log"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
