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
    color1 = [0.6, 0.9, 0.6]
    color2 = [0.6, 0.8, 0.9]
    errorbars=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)

    relu16 = (84.376, 88.624, 89.748)
    nrelu16= (85.832, 88.624, 91.584)
    relu16_std = (1.274962745,  0.9740790522, 1.262287606)
    nrelu16_std = (1.553727775, 0.9740790522, 1.51862767)
    num16 = len(relu16)

    relu32 = (89.458, 90.834, 92.832)
    nrelu32= (89.292, 92.29,  92.084)
    relu32_std = ( 1.3221838, 1.248751376, 0.9020088691)
    nrelu32_std = (0.9496420378, 0.7790699583, 1.167488758)
    num32 = len(relu32)

    relu64 = (92.21,  92.54,  93.414)
    nrelu64= (92.834, 93.034, 94.292)
    relu64_std = (0.746290828,  0.7941977084, 0.2817445652)
    nrelu64_std = (0.7717706913, 0.7235882807, 0.8657771076)
    num64 = len(relu64)
    N = num16+num32+num64
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots(figsize=(N,5))
    b16_relu = ax.bar(ind[:num16], relu16, width, color=color1, yerr=relu16_std, error_kw=errorbars)
    b16_nrelu = ax.bar(ind[:num16] + width, nrelu16, width, color=color2, yerr=nrelu16_std, error_kw=errorbars)
    b32_relu = ax.bar(ind[num16:num16+num32], relu32, width, color=color1, yerr=relu32_std, error_kw=errorbars)
    b32_nrelu = ax.bar(ind[num16:num16+num32] + width, nrelu32, width, color=color2, yerr=nrelu32_std, error_kw=errorbars)
    b64_relu = ax.bar(ind[num16+num32:], relu64, width, color=color1, yerr=relu64_std, error_kw=errorbars)
    b64_nrelu = ax.bar(ind[num16+num32:] + width, nrelu64, width, color=color2, yerr=nrelu64_std, error_kw=errorbars)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy',fontsize=16)
    ax.set_title('NBNN on ' + name + ' - sparse sampling',fontsize=18)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(do_labels(relu16,16) + do_labels(relu32,32) + do_labels(relu64,64), rotation=-30)

    
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    minor_ticks = np.arange(0, 101, 2.5)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True)
    ax.grid(which='minor', alpha=0.4)                                                
    ax.grid(which='major', alpha=0.3)

    for line in gridlines:
        line.set_linestyle('-')
        
    ax.legend((b16_relu[0], b16_nrelu[0]), ('RELU', 'NOT RELU'), loc=4)
    ax.legend((b32_relu[0], b32_nrelu[0]), ('RELU', 'NOT RELU'), loc=4)
    ax.legend((b64_relu[0], b64_nrelu[0]), ('RELU', 'NOT RELU'), loc=4)
    #import pdb; pdb.set_trace()

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height-2.5,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(b16_relu)
    autolabel(b16_nrelu)
    autolabel(b32_relu)
    autolabel(b32_nrelu)
    autolabel(b64_relu)
    autolabel(b64_nrelu)
    
    plt.ylim([min(relu16)-5, max(relu32)+3])
    plt.xlim([ind[0]-0.5, ind[-1]+1])
    plt.tight_layout()
    fig.savefig(name.replace(" ", "")+"_nbnn_sparse.pdf", format="pdf")
    plt.show()