#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":

    sourceT = [85.69, 87.283, 87.925, 88.642, 88, 87.774, 88.415, 88.113, 87.547, 87.358, 88.151]
    source = [0 , 59.932, 60.949, 61.39, 61.153, 61.085, 61.763, 61.966, 61.763, 61.966, 61.424]
    
    err_relu = None
    err_nrelu = None
    
    label_prefix = range(0,110,10)
    labels = [ str(num) + "% retained" for num in label_prefix]
    barPlot_general.do_plot("Amazon to Webcam - Random Fraction of Source Retained", sourceT, source, err_relu, err_nrelu, labels,patch_density="sparse", a_label="Source + Target", b_label="Source only", Y_LIM_MIN=40,ROTATION=15, W_SIZE=12, Y_LIM_MAX=99)
