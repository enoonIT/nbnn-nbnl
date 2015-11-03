#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general

if __name__ == "__main__":
    name = "Sports 8"
    
    ml3_relu = [94.278, 95.292]
    ml3_nrelu = [95.084, 94.292]
    ml3_relu_std = [0.4923108774, 0.619088039]
    ml3_nrelu_std = [0.63468890024, 0.6857258928]

    sm_ml3_relu = [95.286]
    sm_ml3_nrelu= [94.49]
    sm_ml3_relu_std = [0.6860247809]
    sm_ml3_nrelu_std = [0.9890399385]

    res_relu = ml3_relu + sm_ml3_relu
    res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    err_relu = ml3_relu_std + sm_ml3_relu_std
    err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    
    labels = ['32px 3 levels ML3','64px 2 levels ML3','64px 2 levels STOML3']
    barPlot_general.do_plot(name, res_relu, res_nrelu, err_relu, err_nrelu, labels, method="NBNL",W_SIZE=6,MIN_STEP_SIZE=1, ROTATION=10, plot_height=4)
