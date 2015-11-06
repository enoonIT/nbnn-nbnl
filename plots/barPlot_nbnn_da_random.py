#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":



    sourceT = [65.54, 75.608, 76.734, 77.264, 77.511, 77.164, 77.328, 77.356, 77.338, 77.457, 77.136]
    source = [0 , 72.458, 73.357, 73.945, 74.132, 74.025, 74.239, 74.114, 74.203, 74.087, 74.052]
    
    err_relu = None
    err_nrelu = None
    
    label_prefix = range(0,110,10)
    labels = [ str(num) + "% retained" for num in label_prefix]
    barPlot_general.do_plot("Amazon to Caltech - Random Fraction of Source Retained", sourceT, source, err_relu, err_nrelu, labels,patch_density="sparse", a_label="Source + Target", b_label="Source only", Y_LIM_MIN=58,ROTATION=15, W_SIZE=12)
