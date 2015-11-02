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
    color1 = [0.8, 0.5, 0.5]
    color2 = [0.5, 0.5, 0.8]
    errorbars=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)

    relu16 = (39.716,46.034,56.702)
    nrelu16= (45.46,49.12,58.432)
    relu16_std = (1.01954892,0.7891957932,0.3577289477)
    nrelu16_std = (1.396477712,1.138441918,0.8578869389)
    num16 = len(relu16)

    relu32 = (46.97,58.118,63.656)
    nrelu32= (49.998,60.208,63.926)
    relu32_std = (2.004420116,1.009291831, 0.9342537129)
    nrelu32_std = (0.8311257426, 1.397343909, 1.681615295)
    num32 = len(relu32)

    relu64 = (57.672, 63.184)
    nrelu64= (59.748, 63.63)
    relu64_std = (1.278503031, 1.600025)
    nrelu64_std = (1.602473089, 0.9637167634)
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
    ax.set_title('NBNN on ISR 67 - sparse sampling',fontsize=18)
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
    fig.savefig('isr67', dpi=100)
    plt.show()