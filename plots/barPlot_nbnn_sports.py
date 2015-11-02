#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt

def do_labels(elements, size):
    labels = []
    lastS = ''
    for level in range(len(elements)):
        labels.append(str(size) + "px " + str(level+1) + " level"+lastS)
        lastS='s'
    return labels
    
if __name__ == "__main__":
    name = "Sports 8"
    color1 = [0.8, 0.5, 0.5]
    color2 = [0.5, 0.5, 0.8]
    errorbars=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)

    relu16 = (82.46, 87.332, 91.416, 93.332)
    nrelu16= (85.352, 88.836, 94.084, 93.708)
    relu16_std = (1.542741067, 1.321616435, 1.475120334, 1.15890897)
    nrelu16_std = (1.691558453, 1.807478907, 1.037559637, 0.6960747086)
    num16 = len(relu16)

    relu32 = (87.624, 91.708, 93.20)
    nrelu32= (90.874, 93.25, 94.18)
    relu32_std = (1.667597074, 0.8405176976, 1.068770321)
    nrelu32_std = (1.202114803, 1.518354372, 0.8003749122)
    num32 = len(relu32)

    relu64 = (92.374, 94)
    nrelu64= (93.248, 93.666)
    relu64_std = (0.4769486346, 0.9927738917)
    nrelu64_std = (0.7017620679, 1.526689228)
    num64 = len(relu64)
    N = num16+num32+num64
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars

    fig, ax = plt.subplots(figsize=(18,8))
    b16_relu = ax.bar(ind[:num16], relu16, width, color=color1, yerr=relu16_std, error_kw=errorbars)
    b16_nrelu = ax.bar(ind[:num16] + width, nrelu16, width, color=color2, yerr=nrelu16_std, error_kw=errorbars)
    b32_relu = ax.bar(ind[num16:num16+num32], relu32, width, color=color1, yerr=relu32_std, error_kw=errorbars)
    b32_nrelu = ax.bar(ind[num16:num16+num32] + width, nrelu32, width, color=color2, yerr=nrelu32_std, error_kw=errorbars)
    b64_relu = ax.bar(ind[num16+num32:], relu64, width, color=color1, yerr=relu64_std, error_kw=errorbars)
    b64_nrelu = ax.bar(ind[num16+num32:] + width, nrelu64, width, color=color2, yerr=nrelu64_std, error_kw=errorbars)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy',fontsize=16)
    ax.set_title('NBNN on ' + name + ' - dense sampling',fontsize=18)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(do_labels(relu16,16) + do_labels(relu32,32) + do_labels(relu64,64))

    ax.legend((b16_relu[0], b16_nrelu[0]), ('RELU', 'NOT RELU'))
    ax.legend((b32_relu[0], b32_nrelu[0]), ('RELU', 'NOT RELU'))
    ax.legend((b64_relu[0], b64_nrelu[0]), ('RELU', 'NOT RELU'))
    #import pdb; pdb.set_trace()

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.95*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(b16_relu)
    autolabel(b16_nrelu)
    autolabel(b32_relu)
    autolabel(b32_nrelu)
    autolabel(b64_relu)
    autolabel(b64_nrelu)
    
    plt.ylim([min(relu16)-5, max(relu32)+6])
    plt.xlim([ind[0]-0.5, ind[-1]+1])
    fig.savefig(name.replace(" ", ""), dpi=100)
    plt.show()