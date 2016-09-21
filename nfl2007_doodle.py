from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


teams = np.array(['Arizona', 'Atlanta', 'Baltimore', 'Buffalo','Carolina','Chicago', 'Cincinnati','Cleveland','Dallas','Denver','Detroit','Green Bay','Houston','Indianapolis','Jacksonville', 'Kansas City','Miami','Minnesota','N.Y. Giants','N.Y. Jets','New England','New Orleans','Oakland','Philadelphia','Pittsburgh','San Diego','San Francisco','Seattle','St. Louis', 'Tampa Bay','Tennessee', 'Washington'])

wins = np.array([8
,4
,5
,7
,7
,7
,7
,10
,13
,7
,7
,13
,8
,13
,11
,4
,1
,8
,10
,4
,16
,7
,4
,8
,10
,11
,5
,10
,3
,9
,10
,9])

pf = np.array([25.3,16.2,17.2,15.8,16.7,20.9, 23.8,25.1,28.4, 20.0,21.6,27.2,23.7,28.1,25.7, 14.1,16.7,22.8,23.3,16.8,36.8,23.7,17.7,21.0, 24.6,25.8,13.7,24.6, 16.4,20.9,18.8,20.9])

pa = np.array([24.9,25.9,24.0,22.1,21.7,21.8,24.1,23.9, 20.3,25.6,27.8,18.2,24.0,16.4,19.0,20.9,27.3,19.4, 21.9,22.2, 17.1,24.3,24.9,18.8,16.8,17.8,22.8,18.2, 27.4,16.9,18.6,19.4])

data = [pf, pa] # for boxplot
hue_data = (pf/pa)  # for coloring function
hue_std = np.std(hue_data)
hue_mean = np.mean(hue_data)

label_data = zip(teams,pf,pa)  #for labels


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plt.suptitle("2007 NFL")
plt.xlabel("PF")
plt.ylabel("PA")
ax.axhline(np.mean(pa), color='k')
ax.axvline(np.mean(pf), color='k')
ax.set_xlim([np.min(pf)-1, np.max(pf)+1])
ax.set_ylim([np.min(pa)-1, np.max(pa)+1])

#attempt at a colorbar
#cm = plt.cm.get_cmap('gist_rainbow')
#xy = xrange(32)
#z = -1

#

ax.scatter(pf, pa, c=hue_data, cmap='rainbow' ,s=(1+wins*40))



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
