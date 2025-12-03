import pandas as pd
import pathlib
import itertools
import seaborn as sns
import seaborn.objects as so
import matplotlib as mpl

SYNTH_DIR = pathlib.Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/ps_relative_l2_vs_matlab")
OUT_DIR = pathlib.Path("results")
OUT_DIR.mkdir(exist_ok=True)


try:
    df = pd.read_csv(OUT_DIR / "all_loss.csv")
except FileNotFoundError:
    df = []
    for p in SYNTH_DIR.glob("*/*.csv"):
        df.append(pd.read_csv(p))
    df = pd.concat(df).reset_index(drop=True)
    df.to_csv(OUT_DIR / "all_loss.csv", index=False)

# blue and orange, plus reds and greens
optimizer_palette = sns.color_palette("tab10", 2) + sns.color_palette("Reds", 2) + sns.color_palette("Greens", 2)
optimizers = ["Adam", "matlab", "LBFGS-10-6-3", "LBFGS-100-6-3", "LBFGS-10-10-10", "LBFGS-100-10-10"]
optimizer_palette = {k: v for k, v in zip(optimizers, optimizer_palette)}

gb = df.groupby(["optimizer", "img", "device", "synth_iters"])
df = df.set_index(["optimizer", "img", "device", "synth_iters"])
# average time over different seeds, so they can be plotted together (don't have
# capability to put error bars on x-axis)
times = ["synth_only_time"]
for t in times:
    df[f"{t}_mean"] = gb[t].mean()
df = df.reset_index()
display_times = {
    "synth_only_time": "Synthesis duration (s)",
    "total_time": "Total duration (s)",
}
display_funcs = {
    "l2_norm": "L2 Norm",
    "sse": "Sum of Squared Errors",
    "mse": "Mean Squared Error",
}

exclude_opts = ["LBFGS-10-6-3", "LBFGS-100-6-3"]
df = df.query("optimizer not in @exclude_opts")
for n, g in df.groupby("loss_func"):
    height = 5
    loss = g.query("loss_type=='overall'")
    for x in times:
        fig = (
            so.Plot(loss, x=f"{x}_mean", y="loss", color="optimizer", marker="img", linestyle="device")
            .layout(size=(height, height))
            .scale(y="log", color=optimizer_palette, x="log")
            .label(x=display_times[x], y=display_funcs[n])
            .add(so.Line(), so.Agg())
            .add(so.Range(), so.Est(errorbar=("pi", 95)), group='img')
        )
        fig.save(OUT_DIR / f"overall_{x}_{n}.svg", bbox_inches="tight")

    height = 3
    col = "loss_type"
    wrap = 4
    loss = g.query("loss_type!='overall'")
    fig = (
        so.Plot(loss, x=f"{x}_mean", y="loss", color="optimizer", marker="img", linestyle="device")
        .layout(size=(wrap * height, height * df[col].nunique()//wrap))
        .facet(col=col, wrap=wrap)
        .label(x=display_times[x], y=display_funcs[n])
        .scale(color=optimizer_palette, y="log", x="log")
        .share(x=True, y=True)
        .add(so.Line(), so.Agg())
        .add(so.Range(), so.Est(errorbar=("pi", 95)), group='img')
    )
    fig.save(OUT_DIR / f"components_{x}_{n}.svg", bbox_inches="tight")
