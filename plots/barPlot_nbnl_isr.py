#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general
    
if __name__ == "__main__":
    name = "ISR 67"
    ml3_relu = (73.016, 73.046)
    ml3_nrelu = (69.67, 69.99)
    ml3_relu_std = (0.8626297004, 0.3601805103)
    ml3_nrelu_std = (0.8989716347, 0.9401595609)
    num16 = len(ml3_relu)

    sm_ml3_relu = (70.874, 71.532)
    sm_ml3_nrelu= (70.652, 70.862)
    sm_ml3_relu_std = (1.319480959, 1.141564716)
    sm_ml3_nrelu_std = (1.661721998, 0.2845522799)
    num32 = len(sm_ml3_relu)

    res_relu = ml3_relu + sm_ml3_relu
    res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    err_relu = ml3_relu_std + sm_ml3_relu_std
    err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    
    labels = ['32px 3 levels ML3','64px 2 levels ML3','32px 3 levels STOML3', '64px 2 levels STOML3']
    barPlot_general.do_plot(name, res_relu, res_nrelu, err_relu, err_nrelu, labels, method="NBNL",W_SIZE=6,MIN_STEP_SIZE=1, ROTATION=15, plot_height=4.5)