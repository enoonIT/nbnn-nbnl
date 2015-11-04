#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general

def get_sports():
    ml3_relu = [95.292]
    ml3_nrelu = [94.292]
    ml3_relu_std = [0.4923108774]
    ml3_nrelu_std = [0.63468890024]

    sm_ml3_relu = [95.286]
    sm_ml3_nrelu= [94.49]
    sm_ml3_relu_std = [0.6860247809]
    sm_ml3_nrelu_std = [0.9890399385]

    res_relu = ml3_relu + sm_ml3_relu
    res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    err_relu = ml3_relu_std + sm_ml3_relu_std
    err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    labels = ['Sports8 ML3','Sports8 STOML3']
    return [res_relu, res_nrelu, err_relu, err_nrelu, labels]

def get_isr():
    ml3_relu = [73.016]
    ml3_nrelu = [69.67]
    ml3_relu_std = [0.8626297004]
    ml3_nrelu_std = [0.8989716347]

    sm_ml3_relu = [72.632]
    sm_ml3_nrelu= [71.37]
    sm_ml3_relu_std = [0.4538391786]
    sm_ml3_nrelu_std = [1.522793486]

    res_relu = ml3_relu + sm_ml3_relu
    res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    err_relu = ml3_relu_std + sm_ml3_relu_std
    err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    
    labels = ['ISR67 ML3', 'ISR67 STOML3']
    return [res_relu, res_nrelu, err_relu, err_nrelu, labels]
    #return [[],[],[],[],[]]

def get_scenes():
    ml3_relu = [92.092]
    ml3_nrelu = [92.426]
    ml3_relu_std = [0.5764720288]
    ml3_nrelu_std = [0.6440729772]

    sm_ml3_relu = [92.88]
    sm_ml3_nrelu= [92.41]
    sm_ml3_relu_std = [0.8970228537]
    sm_ml3_nrelu_std = [0.5585696018]
    
    res_relu = ml3_relu + sm_ml3_relu
    res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    err_relu = ml3_relu_std + sm_ml3_relu_std
    err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    
    labels = ['Scenes 15 ML3','Scenes 15 STOML3']
    return [res_relu, res_nrelu, err_relu, err_nrelu, labels]
    
#def get_sports():
    #ml3_relu = [94.278, 95.292]
    #ml3_nrelu = [95.084, 94.292]
    #ml3_relu_std = [0.4923108774, 0.619088039]
    #ml3_nrelu_std = [0.63468890024, 0.6857258928]

    #sm_ml3_relu = [95.286]
    #sm_ml3_nrelu= [94.49]
    #sm_ml3_relu_std = [0.6860247809]
    #sm_ml3_nrelu_std = [0.9890399385]

    #res_relu = ml3_relu + sm_ml3_relu
    #res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    #err_relu = ml3_relu_std + sm_ml3_relu_std
    #err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    #labels = ['Sports8 - 32px 3 levels ML3','Sports8 - 64px 2 levels ML3','Sports8 - 64px 2 levels STOML3']
    #return [res_relu, res_nrelu, err_relu, err_nrelu, labels]

#def get_isr():
    #ml3_relu = [73.016, 73.046]
    #ml3_nrelu = [69.67, 69.99]
    #ml3_relu_std = [0.8626297004, 0.3601805103]
    #ml3_nrelu_std = [0.8989716347, 0.9401595609]

    #sm_ml3_relu = [70.874, 71.532]
    #sm_ml3_nrelu= [70.652, 70.862]
    #sm_ml3_relu_std = [1.319480959, 1.141564716]
    #sm_ml3_nrelu_std = [1.661721998, 0.2845522799]

    #res_relu = ml3_relu + sm_ml3_relu
    #res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    #err_relu = ml3_relu_std + sm_ml3_relu_std
    #err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    
    #labels = ['ISR67 - 32px 3 levels ML3','ISR67 - 64px 2 levels ML3','ISR67 - 32px 3 levels STOML3', 'ISR67 - 64px 2 levels STOML3']
    #return [res_relu, res_nrelu, err_relu, err_nrelu, labels]
    
#def get_scenes():
    #ml3_relu = [92.092]
    #ml3_nrelu = [92.426]
    #ml3_relu_std = [0.5764720288]
    #ml3_nrelu_std = [0.6440729772]

    #sm_ml3_relu = [92.88]
    #sm_ml3_nrelu= [92.41]
    #sm_ml3_relu_std = [0.8970228537]
    #sm_ml3_nrelu_std = [0.5585696018]
    
    #res_relu = ml3_relu + sm_ml3_relu
    #res_nrelu = ml3_nrelu + sm_ml3_nrelu
    
    #err_relu = ml3_relu_std + sm_ml3_relu_std
    #err_nrelu = ml3_nrelu_std + sm_ml3_nrelu_std
    
    #labels = ['Scenes 15 - 32px 3 levels ML3','Scenes 15 - 64px 2 levels STOML3']
    #return [res_relu, res_nrelu, err_relu, err_nrelu, labels]

if __name__ == "__main__":
    scenes=get_scenes()
    sports=get_sports()
    isr=get_isr()
    #import pdb; pdb.set_trace()
    seriesA=scenes[0]+sports[0]+isr[0]
    barPlot_general.do_plot("Scenes Datasets", seriesA,  scenes[1]+sports[1]+isr[1],  scenes[2]+sports[2]+isr[2],  scenes[3]+sports[3]+isr[3],  scenes[4]+sports[4]+isr[4], Y_LIM_MIN=40, height_delta=5, ROTATION=20, W_SIZE=8, plot_title="NBNL on Scene Datasets")
