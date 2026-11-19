def status_plot(met):
    kwargs = {
        "included_plots": ["synthesis_imshow", "synthesis_loss", "synthesis_histogram"],
        "synthesis_loss_kwargs": {"plot_penalties": True},
    }
    return po.plot.synthesis_status(met, **kwargs)
