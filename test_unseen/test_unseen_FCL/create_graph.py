#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate
import csv
import pandas as pd

'''data = np.genfromtxt('testdataFCL.csv', delimiter=',', names=['class', 'x', 'y'])
fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.scatter(data['x'], data['y'], c=data['class'])

lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]

ax1.plot(lims, lims, 'k-')

plt.show() # can also do plt.save or something to save the file and then view it later'''

#-----------------------------

'''fig = plt.figure()
ax1 = fig.add_subplot(111)

x = []
y = []
classnum = []

with open('testdataFCL.csv', 'r') as f:
    data = csv.reader(f, delimiter=',')
    for row in data:
        classnum.append(int(row[0]))
        x.append(float(row[1]))
        y.append(float(row[2]))

for i in range(len(classnum)):
    if classnum[i] == 5:
        #fives, = ax1.plot(x[i], y[i], color='#C75FD7', marker='o', label='5')
        fives, = ax1.plot(x[i], y[i], color='#C75FD7', marker='o', markersize=2)
    elif classnum[i] == 6:
        sixes, = ax1.plot(x[i], y[i], color='#5FD7BF', marker='o', markersize=2)
    elif classnum[i] == 9:
        nines, = ax1.plot(x[i], y[i], color='#FFCC99', marker='o', markersize=2)

lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]
reference, = ax1.plot(lims, lims, 'k-')

plt.legend([fives, sixes, nines, reference], ['5', '6', '9', 'Reference Line'])

plt.show()'''

#-------------------

data = np.genfromtxt('testdataFCL.csv', delimiter=',', names=['class', 'x', 'y'])
fig = plt.figure()
ax1 = fig.add_subplot(111)

x = data['x']
y = data['y']
labels = data['class']
df = pd.DataFrame(dict(x = x, y = y, label = labels))

groups = df.groupby('label')

for name, group in groups:
    ax1.plot(group.x, group.y, marker='o', label=name, linestyle='', markersize=2)

lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]
reference, = ax1.plot(lims, lims, 'k-', label='Reference Line')

ax1.legend()

plt.show()
