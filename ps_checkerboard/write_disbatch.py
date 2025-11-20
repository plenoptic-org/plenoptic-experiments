import sys
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_checkerboard")

synth_max_iter= 2000
device= 'cpu'

seed= [0]
line_search_fn= ["strong_wolfe"]
max_iter= [5, 10, 20, 40]
max_eval = [.5, 1, None]
history_size= [30, 100, 130, 200]
lrate = [.1, .3, 1, 3, 10]

prepend = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic OMP_NUM_THREADS=1"

iters = itertools.product(seed, line_search_fn, max_iter, max_eval, history_size, lrate)

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = []
for s, l, it, m, h, lr in iters:
    outdir = f"seed-{s}_search-{l}_iter-{it}_eval-{m}_history-{h}_lr-{lr}"
    if m is not None:
        m = int(m * it)
    cmd = (f"{prepend} python ~/plenoptic_experiments/ps_lbfgs/synthesize.py -s {s} -d {device} "
           f"--max_iter {it} --max_eval {m} --line_search_fn {l} --synth_max_iter {synth_max_iter} "
           f"-o {base_out / outdir} --lr {lr} --history_size {h}")
    cmd = f"({cmd}) &> {base_out / outdir}.log"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
