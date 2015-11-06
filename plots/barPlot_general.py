import numpy as np
import matplotlib.pyplot as plt

def do_labels(elements, sizes):
    labels = []
    for size in sizes:
        lastS = ''
        for level in range(elements):
            labels.append(str(size) + "px " + str(level+1) + " level"+lastS)
            lastS='s'
    return labels

def do_plot(name, res_a, res_b, err_a, err_b, xlabels, a_label="RELU", b_label="NOT RELU", method="NBNN", patch_density="dense", \
            width=0.35, plot_height=5, height_delta=2.5, W_SIZE=None, MIN_STEP_SIZE=2.5,ROTATION=30, Y_LIM_MIN=None, Y_LIM_MAX=None, plot_title=None, \
            add_labels=False):
    color1 = [0.6, 0.9, 0.6]
    color2 = [0.6, 0.8, 0.9]
    errorbars=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)

    N = len(res_a)
    if W_SIZE is None: W_SIZE = N
    ind = np.arange(N)  # the x locations for the groups
           # the width of the bars

    fig, ax = plt.subplots(figsize=(W_SIZE,plot_height))
    series_a = ax.bar(ind, res_a, width, color=color1, yerr=err_a, error_kw=errorbars)
    series_b = ax.bar(ind + width, res_b, width, color=color2, yerr=err_b, error_kw=errorbars)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy',fontsize=16)
    if plot_title is None: plot_title=method + ' on ' + name + ' - ' + patch_density + ' sampling'
    ax.set_title(plot_title,fontsize=18)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(xlabels, rotation=-ROTATION)

    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    minor_ticks = np.arange(0, 101, MIN_STEP_SIZE)
    major_ticks = np.arange(0, 101, 5)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks, minor=False)
    ax.grid(True)
    ax.grid(which='minor', alpha=0.4)
    ax.grid(which='major', alpha=0.2)

    for line in gridlines:
        line.set_linestyle('-')

    ax.legend((series_a[0], series_b[0]), (a_label, b_label), loc=4)

    #import pdb; pdb.set_trace()

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height-height_delta,
                    '%d' % int(height),
                    ha='center', va='bottom')

    if add_labels:
        autolabel(series_a)
        autolabel(series_b)

    if Y_LIM_MIN is None: Y_LIM_MIN=min(min(res_a,res_b))-5
    if Y_LIM_MAX is None: Y_LIM_MAX=max(max(res_a,res_b))+2
    plt.ylim([Y_LIM_MIN, Y_LIM_MAX])
    plt.xlim([ind[0]-0.2, ind[-1]+0.9])
    plt.tight_layout()
    fig.savefig(name.replace(" ", "")+"_" + method + "_" + patch_density + ".pdf", format="pdf")
    plt.show()