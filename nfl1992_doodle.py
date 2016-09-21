from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


teams = np.array(['Atlanta'
,'Buffalo'
,'Chicago'
,'Cincinnati'
,'Cleveland'
,'Dallas'
,'Denver'
,'Detroit'
,'Green Bay'
,'Houston'
,'Indianapolis'
,'Kansas City'
,'LA Raiders'
,'LA Rams'
,'Miami'
,'Minnesota'
,'New England'
,'New Orleans'
,'New York Giants'
,'New York Jets'
,'Philadelphia'
,'Phoenix'
,'Pittsburgh'
,'San Diego'
,'San Francisco'
,'Seattle'
,'Tampa Bay'
,'Washington'])

wins = np.array([6
,11
,5
,5
,7
,13
,8
,5
,9
,10
,9
,10
,7
,6
,11
,11
,2
,12
,6
,4
,11
,4
,11
,11
,14
,2
,5
,9])



pf = np.array([20.4
,23.8
,18.4
,17.1
,17.0
,25.6
,16.4
,17.1
,17.3
,22.0
,13.5
,21.8
,15.6
,19.6
,21.3
,23.4
,12.8
,20.6
,19.1
,13.8
,22.1
,15.2
,18.7
,20.9
,26.9
,8.8
,16.7
,18.8])

pa = np.array([25.9
,17.7
,22.6
,22.8
,17.2
,15.2
,20.6
,20.8
,18.5
,16.1
,18.9
,17.6
,17.6
,23.9
,17.6
,15.6
,22.7
,12.6
,22.9
,19.7
,15.3
,20.8
,14.1
,15.1
,14.8
,19.5
,22.8
,15.9])

data = [pf, pa] # for boxplot
hue_data = (pf/pa) # for coloring function
hue_std = np.std(hue_data)
hue_mean = np.mean(hue_data)
hue_sigma = (hue_data - hue_mean) / hue_std


label_data = zip(teams,pf,pa)  #for labels


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
plt.suptitle("1992 NFL")
plt.xlabel("PF")
plt.ylabel("PA")
ax.axhline(np.mean(pa), color='k')
ax.axvline(np.mean(pf), color='k')
ax.set_xlim([np.min(pf)-1, np.max(pf)+1])
ax.set_ylim([np.min(pa)-1, np.max(pa)+1])


ax.scatter(pf, pa, c=hue_data, s=(1+wins*40))



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
