import pandas as pd
import pathlib
import itertools
import seaborn.objects as so

SYNTH_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/ps_lbfgs_test/initial")
OUT_DIR = pathlib.Path("summary_initial")
OUT_DIR.mkdir(exist_ok=True)


def replace_nan(x, col):
    if pd.isna(x):
        if col == "LBFGS_max_eval":
            return 1.25
        else:
            return "None"
    else:
        return x


def convert_to_sec(x):
    x = x.split(":")
    secs = 0
    for f, s in zip([1, 60, 3600], x[::-1]):
        secs += f * int(s)
    return secs


try:
    df = pd.read_csv(OUT_DIR / "all_loss.csv")
except FileNotFoundError:
    df = []
    for p in SYNTH_DIR.glob("*/loss.csv"):
        df.append(pd.read_csv(p))
    df = pd.concat(df).reset_index()
    df.to_csv(OUT_DIR / "all_loss.csv", index=False)

df.search_func = df.search_func.apply(lambda x: replace_nan(x, "search_func"))
df.LBFGS_max_eval = df.LBFGS_max_eval.apply(lambda x: replace_nan(x, "LBFGS_max_eval"))
df.synth_duration = df.synth_duration.apply(convert_to_sec)

in_plot = "seed"
row = "init_reduced"
col = "coarse_to_fine"
x = "LBFGS_max_iter"
color = "LBFGS_max_eval"
marker = "learning_rate"
across_img = ["filename", "search_func"]
y = ["loss", "synth_duration"]#, "synth_iter"]
y_lims = {
    "loss": [1e-4, 1e-2],
    "synth_duration": [7e2, 1e4],
}
df = df.query("LBFGS_history_size==3 & learning_rate==1")

for indiv in itertools.product(*[df[x].unique() for x in across_img]):
    if not all([isinstance(i, str) for i in indiv]):
        raise ValueError("These need to all be strings!")
    query_str = "&".join([f"{ac}=='{i}'" for ac, i in zip(across_img, indiv)])
    save_str = '_'.join(indiv)
    tmp = df.query(query_str)
    for y_val in y:
        fig = (
            so.Plot(tmp, x=x, y=y_val, color=color, marker=marker)
            .facet(col=col, row=row).label(col=f"ctf: ", row=f"{row}: ")
            .add(so.Dots(), so.Dodge())
            .scale(y="log", x="log", color=so.Nominal())
            .limit(y=y_lims[y_val])
            .share(x=True, y=True)
        )
        fig.save(OUT_DIR / f"{save_str}_{y_val}.svg", bbox_inches="tight")


# for loss:
# - history_size doesn't matter
# - highest max_iter is best, and at that level:
# - with search_func=None, learning_rate 1 clearly better, but much more similar with strong_wolfe
# - search_func, max_eval, ctf and init_reduced don't really matter
#
# for duration
# - learning_rate=1 definitely faster for strong_wolfe, comparable for None
#
# so:
# - history_size=3
# - learning_rate=1 (try higher?)
# - search_func=strong_wolfe
# - max_iter=20 or 40
