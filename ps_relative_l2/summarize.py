import pandas as pd
import pathlib
import itertools
import seaborn.objects as so

SYNTH_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_relative_l2/")
OUT_DIR = pathlib.Path("summary")
OUT_DIR.mkdir(exist_ok=True)


def update_vals(x, col):
    if pd.isna(x[col]):
        if col == "LBFGS_max_eval":
            return 1.25
        else:
            return "None"
    else:
        if col == "LBFGS_max_eval":
            return x[col] / x.LBFGS_max_iter
        else:
            return x[col]


try:
    df = pd.read_csv(OUT_DIR / "all_loss.csv")
except FileNotFoundError:
    df = []
    for p in SYNTH_DIR.glob("*/loss.csv"):
        tmp = pd.read_csv(p)
        # first time we ran this, we forgot to call .item() on the loss before saving,
        # so it ended up as a string of the form "tensor(X, dtype=..., grad_fn=...)". so
        # we need to parse it to a float
        try:
            tmp.loss = tmp.loss.apply(lambda x: float(x.split('(')[1].split(',')[0]))
        except AttributeError:
            # then this is already a float
            pass
        df.append(tmp)
    df = pd.concat(df).reset_index()
    df.to_csv(OUT_DIR / "all_loss.csv", index=False)

df.LBFGS_max_eval = df.apply(lambda x: update_vals(x, "LBFGS_max_eval"), axis=1)

in_plot = ["seed", "filename"]
col = "learning_rate"
marker = "LBFGS_max_iter"
x = "synth_duration"
color = "LBFGS_max_eval"
row = "LBFGS_history_size"
y = "loss"
height = 3
fig = (
    so.Plot(tmp, x=x, y=y, color=color, marker=marker)
    .layout(size=(df[col].nunique() * height, height * df[row].nunique()))
    .facet(col=col, row=row).label(col="lr: ", row=f"history_size: ")
    .add(so.Dots())
    .scale(x="log", color=so.Nominal(), y="log")
    .limit(y=(1e-4, 1e-1))
    .share(x=True, y=True)
)
fig.save(OUT_DIR / f"paired.svg", bbox_inches="tight")
