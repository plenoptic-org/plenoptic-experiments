#!/usr/bin/env python3
import pathlib
import sys

old_dir = pathlib.Path(sys.argv[1])
new_dir = pathlib.Path(sys.argv[2])
out_dir = new_dir.parent / "figs"
out_dir.mkdir(exist_ok=True)

commands = []
for f in old_dir.glob("ps*pt"):
    cmd = f"python plot.py {f} {out_dir / ('old_' + f.with_suffix('.svg').name)}"
    commands.append(cmd)

commands = []
for f in new_dir.glob("ps*pt"):
    cmd = f"python plot.py {f} {out_dir / ('new_' + f.with_suffix('.svg').name)}"
    commands.append(cmd)

with open("disbatch.txt", "w") as f:
    f.write("\n".join(commands))
