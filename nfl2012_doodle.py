from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


teams = np.array(['Arizona', 'Atlanta', 'Baltimore', 'Buffalo','Carolina','Chicago', 'Cincinnati','Cleveland','Dallas','Denver','Detroit','Green Bay','Houston','Indianapolis','Jacksonville', 'Kansas City','Miami','Minnesota','N.Y. Giants','N.Y. Jets','New England','New Orleans','Oakland','Philadelphia','Pittsburgh','San Diego','San Francisco','Seattle','St. Louis', 'Tampa Bay','Tennessee', 'Washington'])


wins = np.array([5
,13
,10
,6
,7
,9
,10
,5
,8
,13
,4
,11
,12
,11
,2
,2
,7
,10
,9
,6
,12
,7
,4
,4
,8
,7
,11
,11
,7
,7
,6
,10])

pf = np.array([15.6
,26.2
,24.9
,21.5
,22.3
,23.4
,24.4
,18.9
,23.5
,30.1
,23.3
,27.1
,26.0
,22.3
,15.9
,13.2
,18.0
,23.7
,26.8
,17.6
,34.8
,28.8
,18.1
,17.5
,21.0
,21.9
,24.8
,25.8
,18.7
,24.3
,20.6
,27.3])

pa = np.array([22.3
,18.7
,21.5
,27.2
,22.7
,17.3
,20.0
,23.0
,25.0
,18.1
,27.3
,21.0
,20.7
,24.2
,27.8
,26.6
,19.8
,21.8
,21.5
,23.4
,20.7
,28.4
,27.7
,27.8
,19.6
,21.9
,17.1
,15.3
,21.8
,24.6
,29.4
,24.3])

data = [pf, pa] # for boxplot
hue_data = (pf/pa)  # for coloring function
hue_std = np.std(hue_data)
hue_mean = np.mean(hue_data)

label_data = zip(teams,pf,pa)  #for labels


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plt.suptitle("2012 NFL")
plt.xlabel("PF")
plt.ylabel("PA")
ax.axhline(np.mean(pa), color='k')
ax.axvline(np.mean(pf), color='k')
ax.set_xlim([np.min(pf)-1, np.max(pf)+1])
ax.set_ylim([np.min(pa)-1, np.max(pa)+1])


ax.scatter(pf, pa, c=hue_data, cmap='jet', s=(1 + (wins * 35)))



#
# for i, team in enumerate(teams):
#     ax.annotate(team, (pf[i],pa[i]), ha = 'right', va = 'bottom')

for label, x, y in zip(teams, pf, pa):
    plt.annotate(label, xy = (x, y), xytext = (-3, 12), textcoords = 'offset points', ha = 'center', va = 'bottom')






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

nygpf = pf[19]
nygpa = pa[19]
nwepf = pf[21]
nwepa = pa[21]
nyg = np.array([pf[19], pa[19]])
nwe = np.array([pf[21], pa[21]])
print 'euclidean_distance = ', np.linalg.norm(nyg-nwe)

nygdelt = nygpf-nygpa
nwedelt = nwepf-nwepa
print 'delt', nygdelt-nwedelt
