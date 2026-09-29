#!/usr/bin/env python3

import pathlib
import pandas as pd
from glob import glob
import altair as alt

BASE_DIR = "/mnt/ceph/users/wbroderick/plenoptic_experiments/penalty_visual_diversity_2/"
df = []
for f in glob(BASE_DIR + "*csv"):
    tmp = pd.read_csv(f)
    tmp["max_iter"] = int(pathlib.Path(f).stem.split("-")[-1])
    df.append(tmp)
df = pd.concat(df)
df.to_csv("dists_metamers.csv")

# encode png image as a base64 string so we can display it,
# see https://altair-viz.github.io/user_guide/marks/image.html
from io import BytesIO
import imageio.v3 as iio
import base64
def encode_base64(x):
    img = iio.imread(x)
    output = BytesIO()
    iio.imwrite(output, img, extension=".png")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()

df["image"] = df.image_path.apply(lambda x: encode_base64(x))

select = alt.selection_point(name="select", on="click", empty=False)
chart = alt.Chart(df).mark_line(point=True).encode(
    x=alt.X("metamer_loss").scale(type="log"),
    y=alt.Y("dists").scale(type="log"),
    color="lr:N",
    shape="max_iter:N",
    tooltip=["obj_func", "metamer_loss", "dists", "lr", "target_image", "loss_func", "penalty_lambda"],
).facet(
    column="target_image:N", row="loss_func"
).add_params(
    select
)

img_faceted = alt.Chart(df, height=256*3.5, width=256*3.5).mark_image().encode(
    url='image'
).facet(
    alt.Facet('image', title='', header=alt.Header(labelFontSize=0))
).transform_filter(
    select
)
(chart | img_faceted).configure(
    autosize=alt.AutoSizeParams(resize=True)
).save(f"dists_results.html")
