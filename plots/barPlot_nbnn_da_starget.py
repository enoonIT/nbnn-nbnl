#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":

    res_relu = [75.3555625, 77.1683125, 83.8320625, 76.4263125, 82.3296875, 86.756, 81.94325, 86.190375, 86.642875]
    res_nrelu = [82.131125, 80.970625, 84.9805, 81.2568125, 84.223125, 87.2415625, 84.19425, 86.9373125, 87.317625]
    
    err_relu = None
    err_nrelu = None
    
    labels = barPlot_general.do_labels(3,[16,32,64])
    barPlot_general.do_plot("Domain Adaptation - Source + Target", res_relu, res_nrelu, err_relu, err_nrelu, labels,patch_density="sparse")
