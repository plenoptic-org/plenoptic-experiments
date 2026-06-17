# Deepnet Metamers

`deepnet_metamers.csv` contains results, which are plotted in `deepnet_metamers_corr.html` and `deepnet_metamers_results.html`

In `deepnet_metamers_corr.html`, we can see that metamer synthesis loss and Pearson Correlation are correlated, so that the lower the loss the higher the correlation (as one would hope). If you hover over a point, you'll see loss, penalty value (encouraging it to lie between 0 and 1), category (all macaw), and pearson correlation value.

In `deepnet_metamers_results.html`, we plot the loss against the learning rate, only for the MSE loss. If you click on a point, it will display the resulting metamer.

## Dependencies

- plenoptic
- polars
- altair
- torchvision
- pandas
