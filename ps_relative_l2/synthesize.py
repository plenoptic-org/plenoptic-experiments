import time
import plenoptic as po
import matplotlib.pyplot as plt
from plenoptic.data.fetch import fetch_data
import torch
import argparse
import pathlib
from typing import Literal
import imageio.v3 as iio
import pandas as pd

def main(
    img: str = "fig4a",
    seed: int = 0,
    max_iter: int = 20,
    max_eval: int | None = None,
    history_size: int = 100,
    device: str | int = 0,
    synth_max_iter: int = 200,
    lr: float = 0.01,
    output_dir: str = "."
):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    fn =  img
    IMG_PATH = fetch_data("portilla_simoncelli_images.tar.gz")
    try:
        device = int(device)
    except:
        pass
    device = torch.device(device)
    n_scales = 3 if img == "fig12b" else 4
    try:
        img = po.load_images(IMG_PATH / f"{img}.jpg")
    except FileNotFoundError:
        img = po.load_images(IMG_PATH / f"{img}.png")
    img = img.to(device).to(torch.float64)
    model = po.simul.PortillaSimoncelli(img.shape[-2:], n_scales=n_scales)
    model.to(device)
    po.tools.set_seed(seed)

    l2_norm = {}
    rep = model.convert_to_dict(model(img))
    for k, v in rep.items():
        l2_norm[k] = torch.linalg.vector_norm(v[~v.isnan()], ord=2)
        l2_norm[k] = l2_norm[k] * torch.ones_like(v)
    l2_norm = model.convert_to_tensor(l2_norm)

    def loss(x, y):
        return po.tools.l2_norm(x / l2_norm, y / l2_norm)

    met = po.synth.Metamer(
        img,
        model,
        loss_function=loss,
    )
    opt_kwargs = {"max_iter": max_iter, "max_eval": max_eval,
                  "history_size": history_size, "line_search_fn": "strong_wolfe",
                  "lr": lr}
    met.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs=opt_kwargs)
    start = time.time()
    met.synthesize(max_iter=synth_max_iter)
    stop = time.time()
    met.save(output_dir / "metamer.pt")
    fig, _ = po.synth.metamer.plot_synthesis_status(met, ylim=False, width_ratios={"plot_representation_error": 3.1})
    fig.savefig(output_dir / f"metamer.svg")
    plt.close(fig)
    iio.imwrite(output_dir / "metamer.png", po.tools.data.convert_float_to_int(po.to_numpy(met.metamer.clip(0, 1)).squeeze()))
    real_loss = po.tools.l2_norm(img, met.metamer).item()
    data = {"filename": fn, "seed": seed, "loss": real_loss,
            "LBFGS_max_iter": max_iter, "LBFGS_max_eval": max_eval,
            "LBFGS_history_size": history_size, "learning_rate": lr, "synth_iter": len(met.losses),
            "synth_loss": met.losses[-1].item(), "synth_duration": stop-start}
    df = pd.DataFrame(data, index=[0])
    df.to_csv(output_dir / "loss.csv", index=False)
    torch.save(met.model(met.metamer), output_dir / "rep.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run some PortillaSimoncelli synthesis to understand LBFGS",
    )
    parser.add_argument("--img", "-i", default="fig4a")
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--max_iter", default=20, type=int)
    parser.add_argument("--max_eval", default=None)
    parser.add_argument("--history_size", default=100, type=int)
    parser.add_argument("--device", "-d", default=0)
    parser.add_argument("--synth_max_iter", default=200, type=int)
    parser.add_argument("--lr", default=.01, type=float)
    parser.add_argument("--output_dir", '-o', default=".")
    args = vars(parser.parse_args())
    for k in ["max_eval"]:
        try:
            args[k] = int(args[k])
        except:
            # then this is None
            args[k] = None
    main(**args)
