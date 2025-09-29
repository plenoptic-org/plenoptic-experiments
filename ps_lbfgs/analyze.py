import sys
import plenoptic as po
import subprocess
import re
import pandas as pd
from plenoptic.data.fetch import fetch_data
import matplotlib.pyplot as plt
import torch
import pathlib
import imageio.v3 as iio

IMG_DIR = fetch_data("portilla_simoncelli_images.tar.gz")
OUTPUT_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_lbfgs/initial")
parse_dir = f"img-([A-z0-9]+)_ctf-([A-z0-9]+)_seed-([A-z0-9]+)_search-([A-z0-9_]+?)_red-([A-z0-9]+)_iter-([A-z0-9]+)_eval-([A-z0-9.]+)_history-([A-z0-9]+)_lr-([A-z0-9.]+)"

if len(sys.argv) > 1:
    directories = [OUTPUT_DIR / f for f in sys.argv[1:]]
else:
    directories = OUTPUT_DIR.iterdir()

for directory in directories:
    if not directory.is_dir():
        continue
    if (directory / "metamer.svg").exists():
        continue
    match = re.findall(parse_dir, directory.name)[0]
    fn, ctf, seed, search, reduced, max_iter, max_eval, history_size, lr = match
    log = directory.with_suffix(f"{directory.suffix}.log")
    if not log.exists():
        raise FileNotFoundError(f"{log=} not found!")
    p = subprocess.Popen(["grep", "-o", "-E", "[0-9:]+<[0-9:]+", log], stdout=subprocess.PIPE)
    time = subprocess.check_output(["tail", "-1"], stdin=p.stdout).decode().split("<")[0]
    try:
        img = po.load_images(IMG_DIR / f"{fn}.jpg")
    except FileNotFoundError:
        img = po.load_images(IMG_DIR / f"{fn}.png")
    img = img.to(torch.float64)
    ps = po.simul.PortillaSimoncelli(img.shape[-2:]).to(torch.float64)
    if eval(ctf):
        met = po.synth.MetamerCTF(
            img,
            ps,
            loss_function=po.tools.optim.l2_norm,
            coarse_to_fine="together",
        )
    else:
        met = po.synth.Metamer(
            img,
            ps,
            loss_function=po.tools.optim.l2_norm,
        )
    old_load = torch.load(directory / "metamer.pt")
    old_load["_current_loss"] = None
    torch.save(old_load, directory / "metamer.pt")
    met.load(directory / "metamer.pt")
    fig, axes = po.synth.metamer.plot_synthesis_status(met, ylim=False)
    fig.savefig(directory / f"metamer.svg")
    iio.imwrite(directory / "metamer.png", po.tools.data.convert_float_to_int(po.to_numpy(met.metamer.clip(0, 1)).squeeze()))
    data = {"filename": fn, "coarse_to_fine": ctf, "seed": seed, "search_func": search, "init_reduced": reduced,
            "LBFGS_max_iter": max_iter, "LBFGS_max_eval": max_eval, "LBFGS_history_size": history_size,
            "learning_rate": lr, "synth_iter": len(met.losses), "loss": met.losses[-1].item(),
            "synth_duration": time}
    df = pd.DataFrame(data, index=[0])
    df.to_csv(directory / "loss.csv", index=False)
    torch.save(met.model(met.metamer), directory / "rep.pt")
    plt.close('all')
