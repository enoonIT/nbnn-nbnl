#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":

    res_relu = [79.36, 85.79425, 86.56925, 86.9140625, 87.0555625, 86.9734375, 87.1341875, 87.0398125, 86.962625, 87.0540625, 86.9443125]
    res_nrelu = [0 , 74.5348125, 75.755625, 75.9544375, 76.165375, 76.2523125, 76.4865, 76.45625, 76.415375, 76.505125, 76.4375]
    
    err_relu = None
    err_nrelu = None
    
    label_prefix = range(0,110,10)
    labels = [ str(num) + "% retained" for num in label_prefix]
    barPlot_general.do_plot("Domain Adaptation - Random Fraction of Source Retained", res_relu, res_nrelu, err_relu, err_nrelu, labels,patch_density="sparse", a_label="Source + Target", b_label="Source only", Y_LIM_MIN=68,ROTATION=15, W_SIZE=12)
