#!/usr/bin/env python3

import pandas as pd
from glob import glob
import altair as alt

BASE_DIR = "/mnt/ceph/users/wbroderick/plenoptic_experiments/fenestration/"
df = pd.concat([pd.read_csv(f) for f in glob(BASE_DIR + "*csv")])
df.to_csv("fenestration_metamers.csv")

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
chart = alt.Chart(df).mark_point().encode(
    x=alt.X("synth_duration"),
    y=alt.Y("loss").scale(type="log"),
    color="lr:N",
    shape="line_search_fn:N",
    tooltip=["loss", "penalty", "line_search_fn", "lr", "max_iter", "history_size"],
).facet(
    column="max_iter:N", row="history_size"
).add_params(
    select
)

img_faceted = alt.Chart(df, height=250, width=250).mark_image().encode(
    url='image'
).facet(
    alt.Facet('image', title='', header=alt.Header(labelFontSize=0))
).transform_filter(
    select
)
(chart | img_faceted).configure(
    autosize=alt.AutoSizeParams(resize=True)
).save(f"fenestration_results.html")
