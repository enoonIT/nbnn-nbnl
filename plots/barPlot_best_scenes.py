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
    name = "comparison"
    color1 = [0.8, 0.5, 0.5]
    color2 = [0.5, 0.5, 0.8]
    errorbars=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)

    ml3_relu = (90, 94.16, 64.62)
    ml3_relu_std = (0.63, 1.13, 1.04)
    ml3_nrelu = (92.88, 95.28, 71.53)
    ml3_nrelu_std = (0.89, 0.61, 0.3)
    num16 = len(ml3_relu)

    N = num16
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars
    
    fig, ax = plt.subplots(figsize=(14,6))
    b16_relu = ax.bar(ind[:num16], ml3_relu, width, color=color1, yerr=ml3_relu_std, error_kw=errorbars)
    b16_nrelu = ax.bar(ind[:num16] + width, ml3_nrelu, width, color=color2, yerr=ml3_nrelu_std, error_kw=errorbars)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy',fontsize=16)
    ax.set_title('Comparison of our method with a linear svm using the whole image',fontsize=18)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(['Scenes 15','Sports 8','ISR 67'])

    ax.legend((b16_relu[0], b16_nrelu[0]), ('Linear SVM', 'NBNL'))
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
    
    plt.ylim([45, max(ml3_relu)+5])
    plt.xlim([ind[0]-0.5, ind[-1]+1])
    fig.savefig(name.replace(" ", "")+"_svm", dpi=100)
    plt.show()