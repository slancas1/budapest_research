#!/usr/bin/env python2.7
# this script reads in EEG data generated by the MindWave
import numpy as np

# loads the raw data (stored in the third column of a CSV file)
def load_data(filename):
	data = np.genfromtxt(filename, delimiter = ',', usecols = (2), skip_header = 1)
	return data

# parameters
stride = 2700
buff_size = 100
sample_size = stride - 2 * buff_size
path = './'
NumFilesPerClass = 3
NumClasses = 2
NumFiles = NumClasses * NumFilesPerClass
NumSamples = 10

# numpy matrices to store the data
splitData = np.zeros([NumFiles, NumSamples, sample_size])
splitLabels = np.zeros([NumFiles, NumSamples])

for i in range(NumFiles):
	if i % NumClasses == 0:
		name = 'math'
	else:
		name = 'relax'

	filenum = int(i / NumClasses) + 1
	print(filenum)
	filename = '{}{}{}.csv'.format(path, name, str(filenum))
	data = load_data(filename)

	idx = 0
	for j in range(NumSamples):
		splitData[i, j, :] = data[idx + buff_size : idx + stride - buff_size]
		idx += stride
		splitLabels[i, j] = i % NumClasses

splitData = np.reshape(splitData, [NumFiles * NumSamples, sample_size])
splitLabels = np.reshape(splitLabels, [NumFiles * NumSamples])

print(splitData.shape)
print(splitLabels)

np.save('math_relax_data', splitData)
np.save('math_relax_labels', splitLabels)
