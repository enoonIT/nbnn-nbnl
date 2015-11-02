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
    name = "Scenes 15"
    color1 = [0.8, 0.5, 0.5]
    color2 = [0.5, 0.5, 0.8]
    errorbars=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)

    ml3_relu = (92.092, 0)
    ml3_nrelu = (92.426, 91.506)
    ml3_relu_std = (0.5764720288, 0.4366119559)
    ml3_nrelu_std = (0.6440729772, 0.0)
    num16 = len(ml3_relu)

    sm_ml3_relu = (92.88)
    sm_ml3_nrelu= (92.41)
    sm_ml3_relu_std = (0.8970228537)
    sm_ml3_nrelu_std = (0.5585696018)
    num32 = 1#len(sm_ml3_relu)

    N = num16+num32
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6))
    b16_relu = ax.bar(ind[:num16], ml3_relu, width, color=color1, yerr=ml3_relu_std, error_kw=errorbars)
    b16_nrelu = ax.bar(ind[:num16] + width, ml3_nrelu, width, color=color2, yerr=ml3_nrelu_std, error_kw=errorbars)
    b32_relu = ax.bar(ind[num16:num16+num32], sm_ml3_relu, width, color=color1, yerr=sm_ml3_relu_std, error_kw=errorbars)
    b32_nrelu = ax.bar(ind[num16:num16+num32] + width, sm_ml3_nrelu, width, color=color2, yerr=sm_ml3_nrelu_std, error_kw=errorbars)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy',fontsize=16)
    ax.set_title('NBNL on ' + name + ' - dense sampling',fontsize=18)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(['32px 3 levels ML3','64px 2 levels ML3','32px 3 levels Smooth ML3'])

    ax.legend((b16_relu[0], b16_nrelu[0]), ('RELU', 'NOT RELU'))
    ax.legend((b32_relu[0], b32_nrelu[0]), ('RELU', 'NOT RELU'))
    #import pdb; pdb.set_trace()

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.96*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(b16_relu)
    autolabel(b16_nrelu)
    autolabel(b32_relu)
    autolabel(b32_nrelu)
    
    plt.ylim([70, max(ml3_relu)+6])
    plt.xlim([ind[0]-0.5, ind[-1]+1])
    fig.savefig(name.replace(" ", "")+"_nbnl", dpi=100)
    plt.show()