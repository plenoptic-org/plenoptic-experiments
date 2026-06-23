#!/usr/bin/env python3

import plenoptic as po
import polars as pl
import glob
import altair as alt
import pathlib
import torch
import numpy as np

BASE_DIR = "/mnt/ceph/users/wbroderick"

df = pl.read_csv(f"{BASE_DIR}/plenoptic_experiments/deepnet_metamer/*csv")
df.write_csv("deepnet_metamers.csv")

# encode png image as a base64 string so we can display it,
# see https://altair-viz.github.io/user_guide/marks/image.html
from io import BytesIO
import imageio.v3 as iio
import base64
def encode_base64(x, replace=("/mnt/ceph/users/wbroderick", BASE_DIR)):
    img = iio.imread(x.replace(*replace).replace(".svg", ".png"))
    output = BytesIO()
    iio.imwrite(output, img, extension=".png")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()

alt.Chart(df).mark_point().encode(
    x=alt.X("loss").scale(type="log"),
    y=alt.Y("pearson_corr").scale(zero=False),
    color="scheduler:N", tooltip=["loss", "penalty", "met_image_category", "pearson_corr"],
).facet(column="loss_func:N", row="layer").save("deepnet_metamers_corr.html")

df = df.with_columns(pl.col("image_path").map_elements(encode_base64, return_dtype=pl.String).alias("image"))

select = alt.selection_point(name="select", on="click", empty=False)
for image in ["parrot", "macaque"]:
    to_plot = df.filter((pl.col("loss_func") == "mse") & (pl.col("image_name") == image))
    chart = alt.Chart(to_plot).mark_point().encode(
        x=alt.X("lr").scale(type="log"),
        y=alt.Y("loss").scale(type="log"),
        # y=alt.Y("pearson_corr").scale(zero=False),
        color="scheduler:N", tooltip=["loss", "penalty", "met_image_category", "pearson_corr"],
    ).facet(column="loss_func:N", row="layer").resolve_scale(y="independent").add_params(
        select
    )

    img_faceted = alt.Chart(to_plot, height=250, width=250).mark_image().encode(
        url='image'
    ).facet(
        alt.Facet('image', title='', header=alt.Header(labelFontSize=0))
    ).transform_filter(
        select
    )
    (chart | img_faceted).configure(
        autosize=alt.AutoSizeParams(resize=True)
    ).save(f"deepnet_metamers_results-{image}.html")
