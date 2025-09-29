import pandas as pd
import pathlib
import itertools
import seaborn.objects as so

SYNTH_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/ps_lbfgs_test/large_lr")
OUT_DIR = pathlib.Path("summary_large_lr")
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

in_plot = ["seed"]
col = "learning_rate"
x = "LBFGS_max_iter"
color = "LBFGS_max_eval"
marker = "filename"
# across_img = ["filename"]
y = ["loss", "synth_duration"]#, "synth_iter"]
scale = {"loss": "log", "synth_duration": None}
for y_val in y:
    if y_sc := scale[y_val]:
        y_sc = {"y": y_sc}
    else:
        y_sc = {}
    fig = (
        so.Plot(df, x=x, y=y_val, color=color, marker=marker)
        .facet(col=col).label(col=f"{col}: ")
        .add(so.Dots(), so.Dodge())
        .scale(x="log", color=so.Nominal(), **y_sc)
        .share(x=True, y=True)
    )
    fig.save(OUT_DIR / f"{y_val}.svg", bbox_inches="tight")
