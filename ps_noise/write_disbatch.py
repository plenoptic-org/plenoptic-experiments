#!/usr/bin/env python3

import sys
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_noise/")

seeds = [0, 1]
figs = ["curie", "einstein"]

device = "cpu"
if len(sys.argv) > 2:
    fn = sys.argv[1]
    device = sys.argv[2]
elif len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = []

synth_iters = [200]
history_size = [100]
weighted = ["1,1,1,1,0,0,100", "10,10,10,1,0,0,100", "10,10,10,10,0,0,100"]
noise = [1, 2]
for f, it, sd, h, wt, n in itertools.product(figs, synth_iters, seeds, history_size, weighted, noise):
    outfile = f"LBFGS-{h}_noise-{n}_wt-{wt}_{device}_{f}_seed-{sd}_iter-{it}"
    cmd = f"bash time_synthesize.sh {f} {n} normal {sd} {h} {device} {it} {wt} {base_out / outfile}.pt"
    cmd = f"({cmd}) &> {base_out / outfile}.log"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
