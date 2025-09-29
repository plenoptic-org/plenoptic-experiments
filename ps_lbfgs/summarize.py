import pandas as pd
import pathlib
import itertools
import seaborn.objects as so

SYNTH_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_lbfgs/history_size")
OUT_DIR = pathlib.Path("summary_history")
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
        df.append(pd.read_csv(p))
    df = pd.concat(df).reset_index()
    df.to_csv(OUT_DIR / "all_loss.csv", index=False)

df.search_func = df.apply(lambda x: update_vals(x, "search_func"), axis=1)
df.LBFGS_max_eval = df.apply(lambda x: update_vals(x, "LBFGS_max_eval"), axis=1)

in_plot = ["seed", "filename"]
col = "learning_rate"
marker = "LBFGS_max_iter"
x = "synth_duration"
color = "LBFGS_max_eval"
row = "LBFGS_history_size"
y = "loss"
scale = {"loss": "log", "synth_duration": None}
height = 3
for f in df.search_func.unique():
    tmp = df.query(f"search_func=='{f}'")
    fig = (
        so.Plot(tmp, x=x, y=y, color=color, marker=marker)
        .layout(size=(df[col].nunique() * height, height * df[row].nunique()))
        .facet(col=col, row=row).label(col="lr: ", row=f"history_size: ")
        .add(so.Dots())
        .scale(x="log", color=so.Nominal(), y="log")
        .limit(y=(1e-4, 1e-1))
        .share(x=True, y=True)
    )
    fig.save(OUT_DIR / f"paired_{f}.svg", bbox_inches="tight")
