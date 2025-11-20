import pandas as pd
import pathlib
import itertools
import seaborn.objects as so

SYNTH_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_checkerboard")
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
        try:
            tmp["coarse_to_fine"]
        except KeyError:
            tmp["coarse_to_fine"] = 0
        df.append(tmp)
    df = pd.concat(df).reset_index()
    df.to_csv(OUT_DIR / "all_loss.csv", index=False)

df.search_func = df.apply(lambda x: update_vals(x, "search_func"), axis=1)
df.LBFGS_max_eval = df.apply(lambda x: update_vals(x, "LBFGS_max_eval"), axis=1)

col = "coarse_to_fine"
marker = "LBFGS_max_iter"
x = "synth_duration"
color = "LBFGS_max_eval"
y = "loss"
scale = {"loss": "log", "synth_duration": None}
height = 3
df = df.query("LBFGS_history_size==100 & learning_rate == 1")
fig = (
    so.Plot(df, x=x, y=y, color=color, marker=marker)
    .layout(size=(df[col].nunique() * height, height))
    .facet(col=col).label(col="ctf: ")
    .add(so.Dots())
    # .scale(x="log", color=so.Nominal(), y="log")
    .scale(color=so.Nominal())
    .share(x=True, y=True)
)
fig.save(OUT_DIR / f"paired.svg", bbox_inches="tight")
