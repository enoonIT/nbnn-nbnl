#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":
    relu16 = (75.44, 80.40, 84.29)
    nrelu16= (77.56, 81.09, 84.83)
    relu16_std = (0.95, 0.67, 1.04)
    nrelu16_std = (0.62, 0.94, 0.97)

    relu32 = (79.106, 84.442, 88.086)
    nrelu32= (79.668, 84.882, 88.254)
    relu32_std = (0.92, 0.44, 1.15)
    nrelu32_std = (0.37, 0.62, 0.26)
	
    relu64 = (82.64, 86.654, 87.132)
    nrelu64= (82.544, 87.278, 87.296)
    relu64_std = (1.18, 0.97, 0.5559856113)
    nrelu64_std = (0.67, 0.36, 0.5444079353)
  
    res_relu = relu16 + relu32 + relu64
    res_nrelu = nrelu16 + nrelu32 + nrelu64
    
    err_relu = relu16_std + relu32_std + relu64_std
    err_nrelu = nrelu16_std + nrelu32_std + nrelu64_std
    
    labels = barPlot_general.do_labels(3,[16,32,64])
    barPlot_general.do_plot("Scenes 15", res_relu, res_nrelu, err_relu, err_nrelu, labels)
