import sys
import itertools
from pathlib import Path

base_out = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_checkerboard_weights")
base_out.mkdir(exist_ok=True)

synth_max_iter= 500
device= 'cpu'

img = ["checkerboard", "fig4a"]
seed= [0]
line_search_fn= ["strong_wolfe"]
max_iter= [10]
max_eval = [1]
history_size= [100]
lrate = [1]
mn = [10]
mag_std = [10]
autocorr_recon = [1., 10, 100]
skew_recon = [0, .1, 1.]
kurt_recon = [0, .1, 1.]
cross_ori = [0, .1, 1.]
cross_scale_mag = [0, .1, 1.]
cross_scale_real = [0, 1., 1]
autocorr_mag = [0, .1, 1.]

prepend = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic OMP_NUM_THREADS=1"

iters = itertools.product(img, seed, line_search_fn, max_iter, max_eval, history_size,
                          lrate, mn, mag_std, autocorr_recon, skew_recon, kurt_recon,
                          cross_ori, cross_scale_mag , cross_scale_real, autocorr_mag,)

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    fn = "disbatch.txt"

commands = []
for im, s, l, it, m, h, lr, m_, st, acr, sr, kr, co, csm, csr, acm in iters:
    outdir = f"img-{im}_seed-{s}_search-{l}_iter-{it}_eval-{m}_history-{h}_lr-{lr}_mn-{m_}_mt-{st}_acr-{acr}_sr-{sr}_kr-{kr}_co-{co}_csm-{csm}_csr-{csr}_acm-{acm}"
    if m is not None:
        m = int(m * it)
    cmd = (f"{prepend} python ~/plenoptic_experiments/ps_checkerboard/synthesize.py -i {im} -s {s} -d {device} "
           f"--max_iter {it} --max_eval {m} --line_search_fn {l} --synth_max_iter {synth_max_iter} "
           f"-o {base_out / outdir} --lr {lr} --history_size {h} --mn {m_} --mag_std {st} --autocorr_recon {acr} "
           f"--skew_recon {sr} --kurt_recon {kr} --cross_ori {co} --cross_scale_mag {csm} --cross_scale_real {csr} "
           f"--autocorr_mag {acm}")
    cmd = f"({cmd}) &> {base_out / outdir}.log"
    commands.append(cmd)

with open(fn, "w") as f:
    f.write('\n'.join(commands))
