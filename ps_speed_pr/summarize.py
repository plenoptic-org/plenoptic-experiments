#!/usr/bin/env python3


import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import pandas as pd
import pathlib


try:
    df = pd.read_csv("all_times.csv")
except FileNotFoundError:
    df = []
    for f in pathlib.Path(".").glob("*npz"):
        timing = np.load(f)
        device, branch = re.findall("ps_speed_timing_([a-z]+)_([a-z_]+).npz", f.name)[0]
        data = {"device": device, "branch": branch}
        for k, v in timing.items():
            if "pyramid" in k:
                func_type = "SteerablePyramid"
                k = k.replace("pyramid_", "")
                k = k.replace("to_tensor_", "").replace("to_pyr_", "")
            elif "ps" in k:
                func_type = "PortillaSimoncelli"
                k = k.replace("ps_", "")
            elif "metamer" in k:
                func_type = "Metamer synthesis (10 iterations)"
                k = k.replace("metamer_", "").replace("synth-10_", "")
            else:
                raise ValueError(f"Don't know how to categorize {k}!")
            data.update({"function": k.replace("_", " "), "time_sec": v.flatten(), "func_type": func_type})
            df.append(pd.DataFrame(data))
    df = pd.concat(df).reset_index(drop=True)
    df.to_csv("all_times.csv", index=False)


for n, g in df.groupby("func_type"):
    f = sns.catplot(
        data=g, x="time_sec", y="device", hue="branch", kind="strip",
        height=2, aspect=4, orient="h", row="function", log_scale=True
    )
    f.fig.suptitle(n, fontsize='xx-large', y=1.1)
    save_n = n.lower().split(' ')[0]
    f.savefig(f"time_{save_n}.png", bbox_inches="tight")
plt.close('all')
