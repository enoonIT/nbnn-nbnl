#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":
    relu16 = (39.716,46.034,56.702)
    nrelu16= (45.46,49.12,58.432)
    relu16_std = (1.01954892,0.7891957932,0.3577289477)
    nrelu16_std = (1.396477712,1.138441918,0.8578869389)

    relu32 = (46.97,58.118,63.656)
    nrelu32= (49.998,60.208,63.926)
    relu32_std = (2.004420116,1.009291831, 0.9342537129)
    nrelu32_std = (0.8311257426, 1.397343909, 1.681615295)

    relu64 = (57.672, 63.184, 62.21)
    nrelu64= (59.748, 63.63, 62.354)
    relu64_std = (1.278503031, 1.600025, 1.29703508)
    nrelu64_std = (1.602473089, 0.9637167634, 1.204753087)

    res_relu = relu16 + relu32 + relu64
    res_nrelu = nrelu16 + nrelu32 + nrelu64
    
    err_relu = relu16_std + relu32_std + relu64_std
    err_nrelu = nrelu16_std + nrelu32_std + nrelu64_std
    labels = barPlot_general.do_labels(3,[16,32,64])
    barPlot_general.do_plot("ISR 67", res_relu, res_nrelu, err_relu, err_nrelu, labels, height_delta=3.5, patch_density="sparse")
