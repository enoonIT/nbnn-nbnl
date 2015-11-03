#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt

color1 = [0.6, 0.9, 0.6]
color2 = [0.6, 0.8, 0.9]
errorbars=dict(ecolor='black', lw=1.5, capsize=3, capthick=1.5)

relu16 = (75.44, 80.40, 84.29, 88.24)
nrelu16= (77.56, 81.09, 84.83, 87.65)
relu16_std = (0.95, 0.67, 1.04, 0.99)
nrelu16_std = (0.62, 0.94, 0.97, 0.72)
num16 = len(relu16)

relu32 = (79.106, 84.442, 88.086)
nrelu32= (79.668, 84.882, 88.254)
relu32_std = (0.92, 0.44, 1.15)
nrelu32_std = (0.37, 0.62, 0.26)
num32 = len(relu32)

relu64 = (82.64, 86.654)
nrelu64= (82.544, 87.278)
relu64_std = (1.18, 0.97)
nrelu64_std = (0.67, 0.36)
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
ax.set_title('NBNN on Scenes 15 - dense sampling',fontsize=18)
ax.set_xticks(ind + width)
ax.set_xticklabels(('16px 1 level', '16px 2 levels', '16px 3 levels', '16px 4 levels', 
                    '32px 1 level', '32px 2 levels', '32px 3 levels',
                    '64px 1 level', '64px 2 levels'))

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

plt.ylim([55.0, 100.0])
plt.xlim([ind[0]-0.5, ind[-1]+1])
plt.show()