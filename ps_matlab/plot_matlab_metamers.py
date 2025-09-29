import plenoptic as po
from plenoptic.data.fetch import fetch_data
import matplotlib.pyplot as plt
import torch
import pathlib

IMG_DIR = fetch_data("portilla_simoncelli_images.tar.gz")
MATLAB_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_matlab")
fn = "fig12a"

for fn in MATLAB_DIR.glob("*.png"):
    fn = fn.stem
    try:
        img = po.load_images(IMG_DIR / f"{fn}.jpg")
    except FileNotFoundError:
        img = po.load_images(IMG_DIR / f"{fn}.png")
    met_img = po.load_images(MATLAB_DIR / f"{fn}.png")
    ps = po.simul.PortillaSimoncelli(img.shape[-2:])
    fig, axes = plt.subplots(1, 3, figsize=(27, 5), gridspec_kw={"width_ratios": [1, 1, 3.1]})
    loss = po.tools.optim.l2_norm(ps(img), ps(met_img))
    for im, ax, t in zip([img, met_img], axes, ["Original image", f"Matlab metamer\n{loss}"]):
        po.imshow(im, ax=ax, title=t)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    ps.plot_representation(ps(img) - ps(met_img), ax=axes[2], ylim=False)
    axes[2].set_title("PS(img) - PS(metamer)", y=1.05)
    fig.savefig(MATLAB_DIR / f"{fn}-metamer.svg")
