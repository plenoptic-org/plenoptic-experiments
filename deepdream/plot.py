def status_plot(met):
    kwargs = {
        "included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"],
        # "synthesis_imshow_kwargs": {"process_image": po.process.rescale},
    }
    fig = po.plot.synthesis_status(met, **kwargs)
    # fig.axes[1].set(yscale="linear")
    return fig
