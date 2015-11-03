#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general

if __name__ == "__main__":
    name = "Scenes 15"
    ml3_relu = [92.092, 0]
    ml3_nrelu = [92.426, 91.506]
    ml3_relu_std = [0.5764720288, 0.4366119559]
    ml3_nrelu_std = [0.6440729772, 0.0]

    sm_ml3_relu = [92.88]
    sm_ml3_nrelu= [92.41]
    sm_ml3_relu_std = [0.8970228537]
    sm_ml3_nrelu_std = [0.5585696018]
    
    res_relu = ml3_relu + sm_ml3_relu
    res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    err_relu = ml3_relu_std + sm_ml3_relu_std
    err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    
    labels = ['32px 3 levels ML3','64px 2 levels ML3','64px 2 levels STOML3']
    barPlot_general.do_plot(name, res_relu, res_nrelu, err_relu, err_nrelu, labels, \
                    method="NBNL",W_SIZE=6,MIN_STEP_SIZE=1, ROTATION=10, Y_LIM_MIN=80, plot_height=4)
