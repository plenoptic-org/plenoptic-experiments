def status_plot(synth):
    kwargs = {
        "included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"],
        "synthesis_loss_kwargs": {"plot_penalties": True},
    }
    fig = po.plot.synthesis_status(synth, **kwargs)
    if synth.loss_function.minmax == "max":
        fig.axes[1].set(yscale="linear")
    return fig
