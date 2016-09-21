from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


teams = np.array(['Arizona', 'Atlanta', 'Baltimore', 'Buffalo','Carolina','Chicago', 'Cincinnati','Cleveland','Dallas','Denver','Detroit','Green Bay','Houston','Indianapolis','Jacksonville', 'Kansas City','Miami','Minnesota','N.Y. Giants','N.Y. Jets','New England','New Orleans','Oakland','Philadelphia','Pittsburgh','San Diego','San Francisco','Seattle','St. Louis', 'Tampa Bay','Tennessee', 'Washington'])


pf = np.array([19.3, 21.5,25.6,21.4,21.2,19.9,22.8, 18.7,29.2,30.1,20.1, 27.7,23.3,28.6,15.6, 22.1,24.3,20.3,23.8, 17.7,27.8,25.1,15.8,29.6,27.3,20.8,19.1,24.6,20.3,17.3,15.9, 18.8])

pa = np.array([18.6,23.4,18.9,18.1,23.4, 27.6,21.5,21.1,22.0,22.1,17.6,19.4,19.2,23.1,25.8,17.6, 23.3,21.4,25.0,25.1,18.7,26.4,28.3,25.0,23.0,20.3,21.3,15.9,22.1,25.6,27.4,27.4])

data = [pf, pa] # for boxplot
hue_data = (pf/pa)*-1  # for coloring function
hue_std = np.std(hue_data)
hue_mean = np.mean(hue_data)

label_data = zip(teams,pf,pa)  #for labels


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plt.suptitle("2014 NFL")
plt.xlabel("PF")
plt.ylabel("PA")
ax.axhline(np.mean(pa), color='k')
ax.axvline(np.mean(pf), color='k')
ax.set_xlim([np.min(pf)-1, np.max(pf)+1])
ax.set_ylim([np.min(pa)-1, np.max(pa)+1])


ax.scatter(pf, pa, c=hue_data, cmap='gist_rainbow', s=(8*30))



#
# for i, team in enumerate(teams):
#     ax.annotate(team, (pf[i],pa[i]), ha = 'right', va = 'bottom')

for label, x, y in zip(teams, pf, pa):
    plt.annotate(label, xy = (x, y), xytext = (-6, 14), textcoords = 'offset points', ha = 'center', va = 'bottom')






"""
#Label method 1
for xy in zip(A, B):
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')



#Label method 2
y=[2.56422, 3.77284,3.52623,3.51468,3.02199]
z=[0.15, 0.3, 0.45, 0.6, 0.75]
n=[58,651,393,203,123]

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i],y[i]))




#label method 3
for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
"""






plt.gca().invert_yaxis()

plt.show()
