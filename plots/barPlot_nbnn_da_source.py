#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":
    res_relu = [50.7543125, 53.3605625, 67.8398125, 52.2821875, 63.5299375, 75.3376875, 63.356125, 73.8944375, 75.28812]
    res_nrelu = [61.245, 59.9325, 70.514375, 60.4951875, 68.63625, 76.915, 68.499375, 76.425625, 77.10687]
    
    err_relu = None
    err_nrelu = None
    
    labels = barPlot_general.do_labels(3,[16,32,64])
    barPlot_general.do_plot("Domain Adaptation - Source Only", res_relu, res_nrelu, err_relu, err_nrelu, labels,patch_density="sparse")
