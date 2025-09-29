import sys
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_lbfgs/history_size")

synth_max_iter= 2000
store_progress= None
device= 'cpu'

img = ["fig4a", "fig16b"]
coarse_to_fine= [False]
seed= [0, 1]
line_search_fn= ["strong_wolfe"]
init_reduced=  [False]
max_iter= [5, 10, 20, 40]
max_eval = [.5, 1, None]
history_size= [200]
lrate = [1]

prepend = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic OMP_NUM_THREADS=1"

iters = itertools.product(img, coarse_to_fine, seed, line_search_fn, init_reduced,
                          max_iter, max_eval, history_size, lrate)

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = []
for i, c, s, l, r, it, m, h, lr in iters:
    outdir = f"img-{i}_ctf-{c}_seed-{s}_search-{l}_red-{r}_iter-{it}_eval-{m}_history-{h}_lr-{lr}"
    if m is not None:
        m = int(m * it)
    cmd = (f"{prepend} python ~/plenoptic_experiments/ps_lbfgs/synthesize.py -i {i} -s {s} -d {device} "
           f"--max_iter {it} --max_eval {m} --line_search_fn {l} --synth_max_iter {synth_max_iter} "
           f"--store_progress {store_progress} -o {base_out / outdir} --lr {lr} --history_size {h}")
    if c:
        cmd += " --coarse_to_fine"
    if r:
        cmd += " --init_reduced"
    cmd = f"({cmd}) &> {base_out / outdir}.log"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
