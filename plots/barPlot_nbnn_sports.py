#!/usr/bin/env python
# a bar plot with errorbars
import barPlot_general


if __name__ == "__main__":
    relu16 = (82.46, 87.332, 91.416)
    nrelu16= (85.352, 88.836, 94.084)
    relu16_std = (1.542741067, 1.321616435, 1.475120334)
    nrelu16_std = (1.691558453, 1.807478907, 1.037559637)
    num16 = len(relu16)

    relu32 = (87.624, 91.708, 93.20)
    nrelu32= (90.874, 93.25, 94.18)
    relu32_std = (1.667597074, 0.8405176976, 1.068770321)
    nrelu32_std = (1.202114803, 1.518354372, 0.8003749122)
    num32 = len(relu32)

    relu64 = (92.374, 94, 93.418)
    nrelu64= (93.248, 93.666, 93.75)
    relu64_std = (0.4769486346, 0.9927738917, 1.386531644)
    nrelu64_std = (0.7017620679, 1.526689228, 0.9672641831)

    res_relu = relu16 + relu32 + relu64
    res_nrelu = nrelu16 + nrelu32 + nrelu64
    
    err_relu = relu16_std + relu32_std + relu64_std
    err_nrelu = nrelu16_std + nrelu32_std + nrelu64_std
    labels = barPlot_general.do_labels(3,[16,32,64])
    barPlot_general.do_plot("Sports 8", res_relu, res_nrelu, err_relu, err_nrelu, labels)
