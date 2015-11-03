#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general
    
if __name__ == "__main__":
    name = "Comparison"

    ml3_relu = (90, 94.16, 64.62)
    ml3_relu_std = (0.63, 1.13, 1.04)
    ml3_nrelu = (92.88, 95.28, 71.53)
    ml3_nrelu_std = (0.89, 0.61, 0.3)
    
    labels = ['Scenes 15','Sports 8','ISR 67']

    labels = ['32px 3 levels ML3','64px 2 levels ML3','64px 2 levels STOML3']
    barPlot_general.do_plot(name, ml3_relu, ml3_nrelu, ml3_relu_std, ml3_nrelu_std, labels, method="comparison",W_SIZE=6,MIN_STEP_SIZE=5, plot_height=4, a_label="Linear SVM",b_label="NBNL", Y_LIM_MIN=40, Y_LIM_MAX=99, height_delta=5, ROTATION=15, plot_title="Method comparison")
