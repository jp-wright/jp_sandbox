from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


teams = np.array(['Arizona', 'Atlanta', 'Baltimore', 'Buffalo','Carolina','Chicago', 'Cincinnati','Cleveland','Dallas','Denver','Detroit','Green Bay','Houston','Indianapolis','Jacksonville', 'Kansas City','Miami','Minnesota','N.Y. Giants','N.Y. Jets','New England','New Orleans','Oakland','Philadelphia','Pittsburgh','San Diego','San Francisco','Seattle','St. Louis', 'Tampa Bay','Tennessee', 'Washington'])


wins = np.array([10
,4
,8
,6
,12
,8
,11
,4
,8
,13
,7
,8
,2
,11
,4
,11
,8
,5
,7
,8
,12
,11
,4
,10
,8
,9
,12
,13
,7
,4
,7
,3])

pf = np.array([23.7
,22.1
,20.0
,21.2
,21.6
,27.8
,26.9
,19.3
,27.4
,37.9
,24.7
,26.1
,17.3
,24.4
,15.4
,26.9
,19.8
,24.4
,18.4
,18.1
,27.8
,25.9
,20.1
,27.6
,23.7
,24.8
,25.4
,26.1
,21.8
,18.0
,22.6
,20.9])

pa = np.array([20.3
,27.7
,22.0
,24.3
,15.1
,29.9
,19.1
,25.4
,27.0
,25.1
,23.5
,26.8
,26.8
,21.0
,28.1
,19.1
,20.8
,30.0
,23.9
,24.2
,21.1
,19.0
,28.3
,23.9
,23.1
,21.8
,17.0
,14.4
,22.8
,24.3
,23.8
,29.9])

data = [pf, pa] # for boxplot
hue_data = (pf/pa)  # for coloring function
hue_std = np.std(hue_data)
hue_mean = np.mean(hue_data)

label_data = zip(teams,pf,pa)  #for labels


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plt.suptitle("2013 NFL")
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
