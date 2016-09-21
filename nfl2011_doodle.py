from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


teams = np.array(['Arizona', 'Atlanta', 'Baltimore', 'Buffalo','Carolina','Chicago', 'Cincinnati','Cleveland','Dallas','Denver','Detroit','Green Bay','Houston','Indianapolis','Jacksonville', 'Kansas City','Miami','Minnesota','N.Y. Giants','N.Y. Jets','New England','New Orleans','Oakland','Philadelphia','Pittsburgh','San Diego','San Francisco','Seattle','St. Louis', 'Tampa Bay','Tennessee', 'Washington'])


wins = np.array([8
,10
,12
,6
,6
,8
,9
,4
,8
,8
,10
,15
,10
,2
,5
,7
,6
,3
,9
,7
,13
,13
,8
,8
,12
,8
,13
,7
,2
,4
,9
,6])

pf = np.array([19.7
,25.1
,23.6
,23.1
,25.4
,22.1
,21.5
,13.6
,23.1
,19.3
,29.6
,35.0
,23.8
,15.2
,15.2
,13.3
,20.6
,21.3
,24.6
,23.6
,32.1
,34.2
,22.4
,24.8
,20.3
,25.4
,23.8
,20.1
,12.1
,17.9
,20.3
,18.0])

pa = np.array([21.6
,21.9
,16.6
,27.3
,26.8
,21.3
,20.2
,19.2
,21.7
,24.4
,24.2
,22.4
,17.4
,26.9
,20.6
,21.1
,19.6
,28.1
,25.0
,22.7
,21.4
,21.2
,27.1
,20.7
,14.2
,23.6
,14.3
,19.7
,25.4
,30.9
,19.8
,22.9])

data = [pf, pa] # for boxplot
hue_data = (pf/pa)*-1  # for coloring function
hue_std = np.std(hue_data)
hue_mean = np.mean(hue_data)

label_data = zip(teams,pf,pa)  #for labels


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plt.suptitle("2011 NFL")
plt.xlabel("PF")
plt.ylabel("PA")
ax.axhline(np.mean(pa), color='k')
ax.axvline(np.mean(pf), color='k')
ax.set_xlim([np.min(pf)-1, np.max(pf)+1])
ax.set_ylim([np.min(pa)-1, np.max(pa)+1])


ax.scatter(pf, pa, c=hue_data, cmap='gist_rainbow', s=(1 + (wins * 35)))



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
