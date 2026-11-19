
import pandas as pd
from pathlib import Path
import altair as alt
from io import BytesIO
import imageio.v3 as iio
import base64
from cyclopts import App, Parameter
from typing import Annotated


app = App()


def create_plot(df, encodings, facets, share_x, share_y, img_size, interactive, output_path, title=None):
    select = alt.selection_point(name="select", on="click", empty=False)
    chart = alt.Chart(df).mark_point().encode(
        **encodings,
    ).add_params(
        select
    )
    if facets:
        chart = chart.facet(**facets)
    if not share_x:
        chart = chart.resolve_scale(x="independent")
    if not share_y:
        chart = chart.resolve_scale(y="independent")
    if title is not None:
        chart = chart.properties(title=title)

    img_faceted = alt.Chart(df, height=img_size, width=img_size).mark_image().encode(
        url='image_vis'
    ).facet(
        alt.Facet('image_vis', title='', header=alt.Header(labelFontSize=0))
    ).transform_filter(
        select
    )
    chart = (chart | img_faceted).configure(
        autosize=alt.AutoSizeParams(resize=True)
    )
    if interactive:
        chart = chart.interactive()
    chart.save(output_path)


@app.default
def main(
    x: str,
    y: str,
    color: str | None = None,
    shape: str | None = None,
    column: str | None = None,
    row: str | None = None,
    tooltip: Annotated[list[str] | None, Parameter(consume_multiple=True)] = None,
    logscale_x: bool = False,
    logscale_y: bool = False,
    logscale_color: bool = False,
    share_x: bool = True,
    share_y: bool = True,
    interactive: bool = False,
    separate_plots: Annotated[list[str] | None, Parameter(consume_multiple=True)] = None,
    img_size: int = 250,
    input_paths: Annotated[list[Path], Parameter(consume_multiple=True)] = [Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/gatys_metamers/run-4")],
    output_path: Path = Path("metamer_results.html")
):
    """Create interactive altair plot.

    This script creates an html page containing an interactive altair plot. Altair is a
    grammar of graphics-inspired library, so if you're familiar with ggplot or seaborn,
    the following syntax should look familiar to you. See their docs for more info:
    https://altair-viz.github.io/

    The plot is a faceted scatter plot designed to summarize info about lots of
    synthesis runs in an easy-to-parse manner. To that end, hovering over the dots will
    present a bunch of info in a tooltip, and clicking on them will show the created
    image.

    Can use the following variables for x, y, color, shape, column, row, tooltip:
    - image_name
    - model
    - loss_function
    - penalty_function
    - initial_image
    - optimizer
    - scheduler
    - penalty_lambda
    - max_iter
    - stop_criterion
    - stop_iters_to_check
    - seed
    - synthesis_loss
    - synthesis_penalty
    - synthesis_duration
    - device
    - dtype

    You will also be able to specify any of the arguments in your optimizer or scheduler
    kwargs dictionary. They will be prefixed with optimizer/scheduler_kwargs. E.g., if
    you passed dictionaries to optimizer_kwargs with the key 'lr', you will be able to
    specify the optimizer_kwargs_lr. View one of you csv files to see the possible
    choices.

    tooltip can be multiple of these (e.g., "--tooltip synthesis_duration dtype"). If
    tooltip is None (the default), will contain all of the above.

    separate_plots specifies which of the above to create separate plots for. it can
    similarly be multiple of these, but if None (the default), will plot all data on
    same plot.

    Parameters
    ----------
    share_x :
        When facetting, whether all x-axes should be shared or not.
    share_y :
        When facetting, whether all y-axes should be shared or not.
    interactive :
        Whether to make the plot interactive or not. This allows you to pan and zoom,
        but seems to make the default x and y limits a little odd.
    img_size :
        Size to display the image at.
    input_paths :
        One of:
        - Directory containing all csvs files summarizing synthesis results. We'll load
          in all csv files contained in *this* directory and its sub-directories
          (only one level, no further).
        - Path to a csv file created by a prior run of this function.
        - Multiple values of the above.
    output_path :
        Path to save the html at. We'll save a csv containing the concatenated results
        in a csv file with the same name.

    """
    df = []
    for inp in input_paths:
        if inp.suffix == ".csv":
            df.append(pd.read_csv(inp))
        else:
            paths = list(inp.glob("*csv")) + list(inp.glob("*/*csv"))
            df.append(pd.concat([pd.read_csv(f) for f in paths
                                 if f != output_path.with_suffix(".csv")]))
    df = pd.concat(df)

    df.to_csv(output_path.with_suffix(".csv"), index=False)

    # image is considered special by altair, so we remove it to prevent issues
    df["image_name"] = df["image"]
    df = df.drop(columns="image")

    if tooltip is None:
        tooltip = [col for col in df.columns if "path" not in col]

    # encode png image as a base64 string so we can display it,
    # see https://altair-viz.github.io/user_guide/marks/image.html
    def encode_base64(x):
        if Path(x).suffix != ".png":
            raise ValueError("Only works for png images!")
        img = iio.imread(x)
        output = BytesIO()
        iio.imwrite(output, img, extension=".png")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


    df["image_vis"] = df.image_path.apply(lambda x: encode_base64(x))


    x = alt.X(x)
    if logscale_x:
        x = x.scale(type="log")
    y = alt.Y(y)
    if logscale_y:
        y = y.scale(type="log")
    encodings = {"x": x, "y": y, "tooltip": tooltip}
    if color is not None:
        color = alt.Color(color)
        if logscale_color:
            color = color.scale(type="log")
        encodings["color"] = color
    elif logscale_color:
        raise ValueError(f"Can't set {logscale_color=} when color is None!")
    if shape is not None:
        encodings["shape"] = alt.Shape(shape, type="nominal")
    facets = {}
    if column is not None:
        facets["column"] = alt.Column(column)
    if row is not None:
        facets["row"] = alt.Row(row)

    if separate_plots is None:
        create_plot(df, encodings, facets, share_x, share_y, img_size, interactive, output_path)
    else:
        for n, g in df.groupby(separate_plots):
            title = [f"{k}-{v}" for k, v in zip(separate_plots, n)]
            title = " ".join(title)
            n = "_".join([str(n_) for n_ in n])
            out = output_path.with_name(f"{output_path.stem}-{n}.html")
            create_plot(g, encodings, facets, share_x, share_y, img_size, interactive, out, title)



if __name__ == "__main__":
    app()
