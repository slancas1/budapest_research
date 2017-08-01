#!/usr/bin/env python2.7

# this is the one-shot code for omniglot data set with LSH implemented

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import datetime
from scipy import misc
import os
import cv2
import math
import itertools
#from tensorflow.python.client import timeline

# set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
#print dt
flags.DEFINE_string('summary_dir', '/tmp/tutorial/{}'.format(dt), 'Summaries directory')

# parameters
BatchLength = 14  # 32 images are in a minibatch
Size = [28, 28, 1]
NumIteration = 100
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
NumSupportsPerClass = 2
NumClassesInSubSet = 2
TrainSize = 6
ValidationSize = 7
TestSize = 7
EvalFreq = 50 # evaluate on every 1000th iteration

if (TrainSize + ValidationSize + TestSize != 20):
	print("DATA NOT PROPERLY SPLIT")
	exit(1)

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]]) # network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2]])
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

# function that finds the number of possible combinations in a given list
def ncr(n, r):
	npr = math.factorial(n) / math.factorial(n - r)
	ncr = npr / math.factorial(r)
	return ncr

# function that makes a list of all of the alphabet / character folders
def make_dir_list(data_dir):
	path_back = "{}/images_background/".format(data_dir)
	path_eval = "{}/images_evaluation/".format(data_dir)
	alphadirs_back = [directory for directory in os.listdir(path_back) if not directory.startswith('.')]
	alphadirs_eval = [directory for directory in os.listdir(path_eval) if not directory.startswith('.')]

	train_datalist = []
	test_datalist = []

	for alphabet in alphadirs_back:
		charpath = "{}{}/".format(path_back, alphabet)
		chardirs = [char for char in os.listdir(charpath) if not char.startswith('.')]
		for character in chardirs:
			train_datalist.append("{}{}/".format(charpath, character))

	for alphabet in alphadirs_eval:
		charpath = "{}{}/".format(path_eval, alphabet)
		chardirs = [char for char in os.listdir(charpath) if not char.startswith('.')]
		for character in chardirs:
			test_datalist.append("{}{}/".format(charpath, character))

	train_datalist = np.asarray(train_datalist)
	test_datalist = np.asarray(test_datalist)

	return train_datalist, test_datalist

# the following code will randomly select five of these directories to use for testing and training
def get_train_data(datalist, train_size = TrainSize, num_classes = NumClasses, Size = [28, 28]):

	#class_nums = random.sample(range(0, len(datalist)), num_classes)
	class_nums = range(num_classes)
	dir_names = datalist[class_nums]

	images = []

	for dir_name in dir_names:
		images.append(['{}{}'.format(dir_name, img) for img in os.listdir(dir_name) if not img.startswith('.')])

	images = np.asarray(images)

	train_set = images[:, 0 : train_size]
	train_set = np.reshape(train_set, num_classes * train_size)

	train_data = np.zeros([train_size * num_classes, Size[0], Size[1]])

	for k in range(train_size * num_classes):
		img = misc.imread(train_set[k])
		train_data[k, :, :] = misc.imresize(img, (Size[0], Size[1]))

	train_labels = np.asarray([int(idx / train_size) for idx in range(train_size * num_classes)])

	permutation = np.random.permutation(train_labels.shape[0])
	train_labels = train_labels[permutation]
	train_data = train_data[permutation]

	return train_data, train_labels

def get_validation_data(datalist, train_size = TrainSize, validation_size = ValidationSize, num_classes = NumClasses, Size = [28, 28]):

	#class_nums = random.sample(range(0, len(datalist)), num_classes)
	class_nums = range(num_classes)
	dir_names = datalist[class_nums]

	images = []

	for dir_name in dir_names:
		images.append(['{}{}'.format(dir_name, img) for img in os.listdir(dir_name) if not img.startswith('.')])

	images = np.asarray(images)

	validation_set = images[:, train_size : train_size + validation_size]
	validation_set = np.reshape(validation_set, num_classes * validation_size)

	validation_data = np.zeros([validation_size * num_classes, Size[0], Size[1]])

	for k in range(validation_size * num_classes):
		img = misc.imread(validation_set[k])
		validation_data[k, :, :] = misc.imresize(img, (Size[0], Size[1]))

	validation_labels = np.asarray([int(idx / validation_size) for idx in range(validation_size * num_classes)])

	permutation = np.random.permutation(validation_labels.shape[0])
	validation_labels = validation_labels[permutation]
	validation_data = validation_data[permutation]

	return validation_data, validation_labels

def get_test_data(datalist, train_size = TrainSize, test_size = TestSize, validation_size = ValidationSize, num_classes = NumClasses, Size = [28, 28]):

	#class_nums = random.sample(range(0, len(datalist)), num_classes)
	class_nums = range(num_classes)

	dir_names = datalist[class_nums]

	images = []

	for dir_name in dir_names:
		images.append(['{}{}'.format(dir_name, img) for img in os.listdir(dir_name) if not img.startswith('.')])

	images = np.asarray(images)

	test_set = images[:, train_size + validation_size : train_size + validation_size + test_size]
	test_set = np.reshape(test_set, num_classes * test_size)

	test_data = np.zeros([test_size * num_classes, Size[0], Size[1]])

	for k in range(test_size * num_classes):
		img = misc.imread(test_set[k])
		test_data[k, :, :] = misc.imresize(img, (Size[0], Size[1]))

	test_labels = np.asarray([int(idx / test_size) for idx in range(test_size * num_classes)])

	permutation = np.random.permutation(test_labels.shape[0])
	test_labels = test_labels[permutation]
	test_data = test_data[permutation]

	return test_data, test_labels

#make the list of extensions to be used by the get data functions
data_location = '../../data'
datalist, _ = make_dir_list(data_location)

#np.random.shuffle(datalist)
datalist = datalist[0 : NumClassesInSubSet]

def all_support_combos(Data, Labels):
	SupportDataList = []
	comb = int(ncr(TrainSize, NumSupportsPerClass))

	for i in range(NumClasses):
		Indices = np.argwhere(Labels == i)
		combos = itertools.combinations(Indices, 2)
		combos = list(combos)
		combos = np.asarray(combos)
		SupportDataList.append([])
		for j in range(comb):
			SupportDataList[i].append(combos[j, :, 0])
	SupportDataList = np.asarray(SupportDataList)

	FullSupportList = []
	for a in SupportDataList[0]:
		for b in SupportDataList[1]:
			FullSupportList.append((a, b))

	FullSupportList = np.asarray(FullSupportList)
	return FullSupportList

def support_index_to_data(Data, Input, class_size = ValidationSize):
	DataList = []
	for ind in Input:
		DataList.append(Data[ind])
	DataList = np.asarray(DataList)
	DataList = np.expand_dims(np.expand_dims(DataList, 4), 0)
	#figure out how to not hard-code 6
	DataTileShape = np.stack([NumClasses * class_size, 1, 1, 1, 1, 1])
	DataList = np.tile(DataList, DataTileShape)
	#print(DataList.shape)

	return DataList

def make_support_set(Data, Labels):
	SupportDataList = np.zeros((BatchLength, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2]))
	QueryData = np.zeros((BatchLength, Size[0], Size[1], Size[2]))
	QueryLabel = np.zeros(BatchLength)

	for i in range(BatchLength):

		QueryClass = np.random.randint(NumClasses)
		QueryIndices = np.argwhere(Labels == QueryClass)
		permutation = np.random.permutation(QueryIndices.shape[0])
		QueryIndices = QueryIndices[permutation]
		SpecificIndex = np.random.randint(QueryIndices.shape[0])
		QueryIndex = QueryIndices[0]


		QueryData[i, :, :, :] = np.reshape(Data[QueryIndex], (Size[0], Size[1], Size[2]))
		QueryLabel[i] = Labels[QueryIndex]

		for j in range(NumClasses):
			if (j == QueryClass):
				for k in range(NumSupportsPerClass):
					if k != SpecificIndex:
						SelectedSupports = Data[QueryIndices[1 + k]]
						SupportDataList[i, k, j, :, :, :] = np.reshape(SelectedSupports, (Size[0], Size[1], Size[2]))
			else:
				SupportIndices = np.argwhere(Labels == j)
				permutation = np.random.permutation(SupportIndices.shape[0])
				SupportIndices = SupportIndices[permutation]
				for k in range(NumSupportsPerClass):
					SelectedSupports = Data[SupportIndices[k]]
					SupportDataList[i, k, j, :, :, :] = np.reshape(SelectedSupports, (Size[0], Size[1], Size[2]))

	return QueryData, SupportDataList, QueryLabel

NumKernels = [16, 16, 16]
def MakeConvNet(Input, Size, First = False):
	CurrentInput = Input
	CurrentInput = (CurrentInput / 255.0) - 0.5
	CurrentFilters = Size[2] # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(len(NumKernels)): # number of layers
		with tf.variable_scope('conv' + str(i)) as varscope:
			if not First:
				varscope.reuse_variables()
			NumKernel = NumKernels[i]
			W = tf.Variable(tf.random_normal([3, 3, CurrentFilters, NumKernel], stddev = 0.1), name = "W")
			#W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel]) # this should be 3 and 3 if it is CNN friendly
			#Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.1))

			CurrentFilters = NumKernel
			ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'VALID') #VALID, SAME
			#ConvResult= tf.add(ConvResult, Bias)

			# ReLU = tf.nn.relu(ConvResult)

			# leaky ReLU
			alpha = 0.01
			ReLU = tf.maximum(alpha * ConvResult, ConvResult)

			CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'VALID') # this should be 1, 1, 1, 1 for both if the network is CNN friendly

	CurrentInput = tf.nn.dropout(CurrentInput, KeepProb)

	return CurrentInput

with tf.name_scope('network'):
	with tf.name_scope('query_network'):
		encodedQuery = MakeConvNet(InputData, Size, First = True)
	SupportList = []
	QueryList = []

	with tf.name_scope('supports_network'):
		for i in range(NumClasses):
			for k in range(NumSupportsPerClass):
				SupportList.append(MakeConvNet(SupportData[:, k, i, :, :, :], Size))
				QueryList.append(encodedQuery)

		QueryRepeat = tf.stack(QueryList)
		Supports = tf.stack(SupportList)

with tf.name_scope('cosine_similarity'):
	'''calculate cosine similarity between encodedQuery and everything in Supports
	(A*B)/(|A||B|)'''

	DotProduct = tf.reduce_sum(tf.multiply(QueryRepeat, Supports), [2, 3, 4])
	#QueryMag = tf.sqrt(tf.reduce_sum(tf.square(QueryRepeat), [2, 3, 4]))
	SupportsMag = tf.sqrt(tf.reduce_sum(tf.square(Supports), [2, 3, 4]))
	CosSim = DotProduct / tf.clip_by_value(SupportsMag, 1e-10, float("inf"))

	CosSim = tf.reshape(CosSim, [NumClasses, NumSupportsPerClass, -1])
	CosSim = tf.transpose(tf.reduce_sum(CosSim, 1))

	probs = tf.nn.softmax(CosSim)

	CosSim = tf.reduce_mean(tf.losses.softmax_cross_entropy(OneHotLabels, CosSim))

with tf.name_scope('lsh'):
	OutputDimension = 10
	DistanceMetric = 'Euclidean' # Hamming, Euclidean, EucalideanSquared, EuclideanCentered, L1Norm, Cosine
	QueryRepeat = tf.transpose(QueryRepeat, [1, 0, 2, 3, 4])
	QueryShape = QueryRepeat.shape
	QueryRepeat = tf.reshape(QueryRepeat, [-1, int(QueryShape[1]), int(QueryShape[2]) * int(QueryShape[3]) * int(QueryShape[4])])
	QueryRepeat = tf.expand_dims(QueryRepeat, 3)
	#QueryRepeat = tf.tile(QueryRepeat, [1, 1, 1, OutputDimension])

	Supports = tf.transpose(Supports, [1, 0, 2, 3, 4])
	SupportsShape = Supports.shape
	Supports = tf.reshape(Supports, [-1, int(SupportsShape[1]), int(SupportsShape[2]) * int(SupportsShape[3]) * int(SupportsShape[4])])
	Supports = tf.expand_dims(Supports, 3)
	#Supports = tf.tile(Supports, [1, 1, 1, OutputDimension])

	InputDimension = Supports.shape[2]
	NumStoredValues = Supports.shape[1] # number of stored values is the same as the number of supports

	with tf.name_scope('plane_projection'):
		RandomPlanes = tf.Variable(tf.random_normal([1, 1, int(OutputDimension), int(InputDimension)], mean = 0, stddev = 1), name = "Planes")
		RandomPlanes = tf.nn.dropout(RandomPlanes, KeepProb)
		RandomPlanesM = tf.tile(RandomPlanes, [int(BatchLength), int(NumStoredValues), 1, 1])

		with tf.name_scope('projected_supports'):
			# proejct them based on the random planes
			ProjectedValues = tf.matmul(RandomPlanesM, Supports) # how does this multiplication result in a distance??
			ProjectedValues = tf.squeeze(ProjectedValues)

		with tf.name_scope('projected_query'):
			ProjectedQuery = tf.matmul(RandomPlanesM, QueryRepeat)
			ProjectedQuery = tf.squeeze(ProjectedQuery)

	# decide if projection is above or below the plane
	# create a constant zero map for comparison
	Comp = tf.constant(np.zeros((BatchLength, NumStoredValues, OutputDimension)), dtype = tf.float32)

	with tf.name_scope('hashing'):
		with tf.name_scope('hashed_query'):
			HashQuery = tf.cast(tf.greater(ProjectedQuery, Comp)[:, :, :], tf.int32)
		with tf.name_scope('hashed_supports'):
			HashSupports = tf.cast(tf.greater(ProjectedValues, Comp)[:, :, :], tf.int32)

	with tf.name_scope('distance_calc'):
		#calculate distance between Query and hashed values
		if DistanceMetric == 'Hamming':
			LogXOR = tf.logical_xor(tf.cast(HashQuery, tf.bool), tf.cast(HashSupports, tf.bool))
			Dist = tf.reduce_sum(tf.cast(LogXOR, tf.int32), 2)
			SelectedInd = tf.argmin(Dist, 1)
			#RetrievedElement = ValuesToHash[tf.cast(SelectedInd, tf.int32), :, :]
		if DistanceMetric == 'Euclidean':
			diff = tf.subtract(tf.cast(HashQuery, tf.float32), tf.cast(HashSupports, tf.float32))
			Dist = tf.sqrt(tf.square(tf.reduce_sum(diff, 2)))
			SelectedInd = tf.argmin(Dist, 1)
			#RetrievedElement = ValuesToHash[tf.cast(SelectedInd, tf.int32), :, :]
		if DistanceMetric == 'EuclideanCentered':
			# This is not the same as in the library, but we should check if it is ok
			X = tf.cast(HashQuery, tf.float32)
			X = X - tf.reduce_mean(X)
			Y = tf.cast(HashSupports, tf.float32)
			Y = Y - tf.reduce_mean(Y)
			diff = tf.subtract(X, Y)
			Dist = tf.sqrt(tf.square(tf.reduce_sum(diff, 1)))
			SelectedInd = tf.argmin(Dist, 1)
			#RetrievedElement = ValuesToHash[tf.cast(SelectedInd, tf.int32), :, :]
		if DistanceMetric == 'EucalideanSquared':
			diff = tf.subtract(tf.cast(HashQuery, tf.float32), tf.cast(HashSupports, tf.float32))
			# we do not need the square of the distances because the square of zeros and ones remains the same
			Dist = tf.reduce_sum(tf.square(diff), 2)
			SelectedInd = tf.argmin(Dist, 1)
			#RetrievedElement = ValuesToHash[tf.cast(SelectedInd, tf.int32), :, :]
		if DistanceMetric == 'L1Norm':
			Dist = tf.reduce_sum(tf.abs(tf.subtract(HashQuery, HashSupports)), 2)
			SelectedInd = tf.argmin(Dist, 1)
			#RetrievedElement = ValuesToHash[tf.cast(SelectedInd, tf.int32), :, :]
		if DistanceMetric == 'Cosine':
			DotProduct = tf.reduce_sum(tf.multiply(tf.cast(HashQuery, tf.float32), tf.cast(HashSupports, tf.float32)), 2) # necessary b/c actually -1s
			MagX = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(HashQuery, tf.float32)), 2))
			MagY = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(HashSupports, tf.float32)), 2))
			Dist = DotProduct / (tf.multiply(MagX, MagY))
			#cosine similarity selects the largest value!
			SelectedInd = tf.argmax(Dist, 1)
			#RetrievedElement = ValuesToHash[tf.cast(SelectedInd, tf.int32), :, :]

		SelectedInd = tf.cast(tf.cast(SelectedInd/NumClasses, tf.int32), tf.float64)

# define loss and optimizer
with tf.name_scope('loss'):
	def CalculateDistance(Vector1, Vector2, Comp):
		if DistanceMetric == 'Euclidean' or DistanceMetric == 'EucalideanSquared':
			Diff1 = tf.square(tf.subtract(tf.cast(Vector1, tf.float32), tf.cast(Comp, tf.float32)))
			Diff2 = tf.square(tf.subtract(tf.cast(Vector2, tf.float32), tf.cast(Comp, tf.float32)))
		if DistanceMetric == 'L1Norm':
			Diff1 = tf.abs(tf.subtract(tf.cast(Vector1, tf.float32), tf.cast(Comp, tf.float32)))
			Diff2 = tf.abs(tf.subtract(tf.cast(Vector2, tf.float32), tf.cast(Comp, tf.float32)))
		Diff = tf.add(Diff1, Diff2)
		return Diff
	#use only those differences where the sign is different
	def InGroupSignedDif(Vector1, Vector2):
		Comp = tf.constant(np.zeros((BatchLength, OutputDimension)), dtype = tf.float32)
		Diff = CalculateDistance(Vector1, Vector2, Comp)

		FirstLarger = tf.greater(Vector1, Comp)
		SecondSmaller = tf.greater(Comp, Vector2)
		SignDifferences1 = tf.logical_and(FirstLarger, SecondSmaller)
		#OR
		SecondLarger = tf.greater(Vector2, Comp)
		FisrtSmaller = tf.greater(Comp, Vector1)
		SignDifferences2 = tf.logical_and(SecondLarger, FisrtSmaller)
		SignDifferences = tf.logical_or(SignDifferences1, SignDifferences2)
		SignSimilartiy = tf.cast(tf.logical_not(SignDifferences), tf.float32)
		SignDifferences = tf.cast(SignDifferences, tf.float32)
		DiffToDecrease = tf.multiply(Diff, SignDifferences)
		DiffToIncrease = tf.multiply(Diff, SignSimilartiy)
		DiffToIncrease = tf.clip_by_value(tf.multiply(Diff, SignSimilartiy), 0, 1)

		DiffToDecrease = tf.reduce_mean(DiffToDecrease, 1)
		DiffToIncrease = tf.reduce_mean(DiffToIncrease, 1)
		if DistanceMetric == 'Euclidean':
			DiffToDecrease = tf.maximum(DiffToDecrease, 1e-15)
			DiffToIncrease = tf.maximum(DiffToIncrease, 1e-15)
			DiffToDecrease = tf.sqrt(DiffToDecrease)
			DiffToIncrease = tf.sqrt(DiffToIncrease)
		#TotDiff=tf.divide(DiffToDecrease,DiffToIncrease)
		#TotDiff=tf.subtract((10*DiffToDecrease)*(10*DiffToDecrease),DiffToIncrease)
		TotDiff = tf.subtract(DiffToDecrease, DiffToIncrease)
		return TotDiff

	def CrossGroupSignedDif(Vector1, Vector2):
		Comp = tf.constant(np.zeros((BatchLength, OutputDimension)), dtype = tf.float32)
		Diff = CalculateDistance(Vector1, Vector2, Comp)

		FirstLarger = tf.greater(Vector1, Comp)
		SecondLarger = tf.greater(Vector2, Comp)
		SignSimmilarity1 = tf.logical_and(FirstLarger, SecondLarger)
		#OR
		FisrtSmaller = tf.greater(Comp, Vector1)
		SecondSmaller = tf.greater(Comp, Vector2)
		SignSimmilarity2 = tf.logical_and(FisrtSmaller, SecondSmaller)
		SignSimmilarity = tf.logical_or(SignSimmilarity1, SignSimmilarity2)
		SignDifferences = tf.cast(tf.logical_not(SignSimmilarity), tf.float32)
		SignSimmilarity = tf.cast(SignSimmilarity, tf.float32)
		DiffToDecrease = tf.multiply(Diff, SignSimmilarity)
		DiffToIncrease = tf.multiply(Diff, SignDifferences)
		DiffToIncrease = tf.clip_by_value(tf.multiply(Diff, SignDifferences), 0, 1)

		DiffToDecrease = tf.reduce_mean(DiffToDecrease, 1)
		DiffToIncrease = tf.reduce_mean(DiffToIncrease, 1)
		if DistanceMetric == 'Euclidean':
			DiffToDecrease = tf.maximum(DiffToDecrease, 1e-15)
			DiffToIncrease = tf.maximum(DiffToIncrease, 1e-15)
			DiffToDecrease = tf.sqrt(DiffToDecrease)
			DiffToIncrease = tf.sqrt(DiffToIncrease)
		#TotDiff=tf.divide(DiffToDecrease,DiffToIncrease)
		#TotDiff=tf.subtract((10*DiffToDecrease)*(10*DiffToDecrease),DiffToIncrease)
		TotDiff = tf.subtract(DiffToDecrease, DiffToIncrease)
		return TotDiff

	def QuerryDifNotSame(SelectedQuery, ProjectedValues):
		Comp = tf.constant(np.zeros((BatchLength, (NumClasses * NumSupportsPerClass), OutputDimension)), dtype = tf.float32)
		Diff = CalculateDistance(SelectedQuery, ProjectedValues, Comp)

		FirstLarger = tf.greater(SelectedQuery, Comp)
		SecondLarger = tf.greater(ProjectedValues, Comp)
		SignSimmilarity1 = tf.logical_and(FirstLarger, SecondLarger)
		#OR
		FisrtSmaller = tf.greater(Comp, SelectedQuery)
		SecondSmaller = tf.greater(Comp, ProjectedValues)
		SignSimmilarity2 = tf.logical_and(FisrtSmaller, SecondSmaller)
		SignSimmilarity = tf.logical_or(SignSimmilarity1, SignSimmilarity2)
		SignDifferences = tf.cast(tf.logical_not(SignSimmilarity), tf.float32)
		SignSimmilarity = tf.cast(SignSimmilarity, tf.float32)
		DiffToDecrease = tf.multiply(Diff, SignSimmilarity)
		DiffToIncrease = tf.multiply(Diff, SignDifferences)
		DiffToIncrease = tf.clip_by_value(tf.multiply(Diff, SignDifferences), 0, 1)

		DiffToDecrease = tf.reduce_mean(DiffToDecrease, 2)
		DiffToIncrease = tf.reduce_mean(DiffToIncrease, 2)
		if DistanceMetric == 'Euclidean':
			DiffToDecrease = tf.maximum(DiffToDecrease, 1e-15)
			DiffToIncrease = tf.maximum(DiffToIncrease, 1e-15)
			DiffToDecrease = tf.sqrt(DiffToDecrease)
			DiffToIncrease = tf.sqrt(DiffToIncrease)
		#TotDiff=tf.divide(DiffToDecrease,DiffToIncrease)
		TotDiff = tf.subtract(DiffToDecrease, DiffToIncrease)
		return TotDiff

	def QuerryDifSame(SelectedQuery, ProjectedValues):
		Comp = tf.constant(np.zeros((BatchLength, (NumClasses * NumSupportsPerClass), OutputDimension)), dtype = tf.float32)
		Diff = CalculateDistance(SelectedQuery, ProjectedValues, Comp)

		FirstLarger = tf.greater(SelectedQuery, Comp)
		SecondSmaller = tf.greater(Comp, ProjectedValues)
		SignDifferences1 = tf.logical_and(FirstLarger, SecondSmaller)
		#OR
		SecondLarger = tf.greater(ProjectedValues, Comp)
		FisrtSmaller = tf.greater(Comp, SelectedQuery)
		SignDifferences2 = tf.logical_and(SecondLarger, FisrtSmaller)
		SignDifferences = tf.logical_or(SignDifferences1, SignDifferences2)
		SignSimilartiy = tf.cast(tf.logical_not(SignDifferences), tf.float32)
		SignDifferences = tf.cast(SignDifferences, tf.float32)
		DiffToDecrease = tf.multiply(Diff, SignDifferences)
		DiffToIncrease = tf.multiply(Diff, SignSimilartiy)
		DiffToIncrease = tf.clip_by_value(tf.multiply(Diff, SignSimilartiy), 0, 1)

		DiffToDecrease = tf.reduce_mean(DiffToDecrease, 2)
		DiffToIncrease = tf.reduce_mean(DiffToIncrease, 2)

		if DistanceMetric == 'Euclidean':
			DiffToDecrease = tf.maximum(DiffToDecrease, 1e-15)
			DiffToIncrease = tf.maximum(DiffToIncrease, 1e-15)
			DiffToDecrease = tf.sqrt(DiffToDecrease)
			DiffToIncrease = tf.sqrt(DiffToIncrease)
		#TotDiff=tf.divide(DiffToDecrease,DiffToIncrease)
		TotDiff = tf.subtract(DiffToDecrease, DiffToIncrease)
		return TotDiff

	Signs = tf.stack([OneHotLabels[:, 0], OneHotLabels[:, 0], OneHotLabels[:, 1], OneHotLabels[:, 1]])
	Signs = tf.transpose(Signs, [1, 0])
	Signs = tf.expand_dims(Signs, -1)
	Signs = tf.tile(Signs, [1, 1, OutputDimension])
	AntiSigns = 1 - Signs
	ToComp = tf.multiply(Signs, ProjectedQuery)
	ToComp = tf.add(ToComp, tf.multiply(AntiSigns, ProjectedValues))
	#positive
	SameDif = QuerryDifSame(ToComp, ProjectedValues)
	ToComp = tf.multiply(AntiSigns, ProjectedQuery)
	ToComp = tf.add(ToComp, tf.multiply(Signs, ProjectedValues))
	NotSameDif = QuerryDifNotSame(ToComp, ProjectedValues)
	LossQ = tf.reduce_mean(tf.add(SameDif, NotSameDif))

	NumInGroup = ncr(NumSupportsPerClass, 2) * NumClasses
	NumCrossGroup = ncr((NumSupportsPerClass * NumClasses), 2) - NumInGroup

	count = 0
	for Class in range(0, NumClasses):
		for i in range(NumSupportsPerClass - 1):
			for j in range(i + 1, NumSupportsPerClass):
				InGroupDiffElement = InGroupSignedDif(ProjectedValues[:, i + (Class * NumSupportsPerClass), :], ProjectedValues[:, j + (Class * NumSupportsPerClass), :])
				if count == 0:
					InGroupDiff = InGroupDiffElement
				else:
					InGroupDiff = tf.add(InGroupDiff, InGroupDiffElement)
				count += 1

	count = 0
	for Class1 in range(0, NumClasses - 1):
		for Class2 in range(Class1 + 1, NumClasses):
			for k in range(NumSupportsPerClass):
				for m in range(NumSupportsPerClass):
					CrossGroupDiffElement = CrossGroupSignedDif(ProjectedValues[:, k + (Class1 * NumSupportsPerClass), :], ProjectedValues[:, m + (Class2 * NumSupportsPerClass), :])
					if count == 0:
						CrossGroupDiff = CrossGroupDiffElement
					else:
						CrossGroupDiff = tf.add(CrossGroupDiff, CrossGroupDiffElement)
					count += 1

	InGroupDiff = tf.reduce_mean(InGroupDiff)
	CrossGroupDiff = tf.reduce_mean(CrossGroupDiff)
	#Loss=tf.divide(InGroupDiff,CrossGroupDiff)
	#Loss=LossQ
	Loss = tf.add(tf.add(InGroupDiff, 0.5 * CrossGroupDiff), 10 * LossQ)

ConvVarList = []
for v in tf.global_variables():
	if ('Planes' not in v.name) and ('loss' not in v.name):
		ConvVarList.append(v)

PlaneVarList = []
for v in tf.global_variables():
	if ('Planes'  in v.name):
		PlaneVarList.append(v)

with tf.name_scope('optimizer'):
		# use ADAM optimizer this is currently the best performing training algorithm in most cases
		#Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
		Optimizer1 = tf.train.GradientDescentOptimizer(1e-4).minimize(Loss, var_list = ConvVarList)
		Optimizer2 = tf.train.GradientDescentOptimizer(1e-6).minimize(Loss, var_list = PlaneVarList)
		#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
		Correct = tf.cast(InputLabels, tf.float64)
		CorrectPredictions = tf.equal(SelectedInd, Correct)
		Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))

# initializing the variables
Init = tf.global_variables_initializer()

# create sumamries, these will be shown on tensorboard

# histogram sumamries about the distributio nof the variables
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)

# create image summary from the first 10 images
#tf.summary.image('images', TrainData[1 : 10, :, :, :], max_outputs = 50)

# create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
tf.summary.scalar("accuracy", Accuracy)

SummaryOp = tf.summary.merge_all()

# limits the amount of GPU you can use so you don't tie up the server
conf = tf.ConfigProto(allow_soft_placement = True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.25

# launch the session with default graph
with tf.Session(config = conf) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir, tf.get_default_graph())
	Saver = tf.train.Saver()
	runOptions = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
	runMetaData = tf.RunMetadata()

	# keep training until reach max iterations - other stopping criterion could be added
	for Step in range(1, NumIteration + 1):
			TrainData, TrainLabels = get_train_data(datalist)
			# need to change make_support_set to create all combinations
			# need to calculate the accuracy for each combination and keep track of which pair of supports produces the max accuracy
			QueryData, SupportDataList, Label = make_support_set(TrainData, TrainLabels)

			# execute teh session
			Summary, _, _, CS, Acc, L,  c, cp, IGD, CGD = Sess.run([SummaryOp, Optimizer1, Optimizer2, CosSim, Accuracy, Loss, Correct, SelectedInd, InGroupDiff, CrossGroupDiff],
				options = runOptions, run_metadata = runMetaData,
				feed_dict = {InputData: QueryData, InputLabels: Label, SupportData: SupportDataList, KeepProb: 0.8})

			SummaryWriter.add_run_metadata(runMetaData, 'step%d' % Step)
			SummaryWriter.add_summary(Summary, Step)

			'''tl = timeline.Timeline(runMetaData.step_stats)
			ctf = tl.generate_chrome_trace_format()
			with open('timeline.json', 'w') as f:
				f.write(ctf)'''

			if (Step % 50 == 0):
				print("Iteration: " + str(Step))
				print("Accuracy: " + str(Acc))
				print("Loss: " + str(L))
				#print("In Group Difference: " + str(IGD))
				#print("Cross Group Difference: " + str(CGD))

			# independent test accuracy
			if not Step % EvalFreq:
				TotalAcc = 0
				count = 0
				for i in range(BatchLength):
					TestData, TestLabels = get_test_data(datalist)
					TestData, SuppData, TestLabels = make_support_set(TestData, TestLabels)

					Acc, L, c, Ind = Sess.run([Accuracy, Loss, SelectedInd, Correct],
						feed_dict = {InputData: TestData, InputLabels: TestLabels, SupportData: SuppData, KeepProb: 1.0})
					TotalAcc += Acc
					count += 1
				TotalAcc = TotalAcc / count
				print("Independent Test set: ", TotalAcc)
			SummaryWriter.add_summary(Summary, Step)

	print("Finding best supports...")

	AllCombos = all_support_combos(TrainData, TrainLabels)
	ValidationData, ValidationLabels = get_validation_data(datalist)
	ValidationData = np.reshape(ValidationData, [ValidationData.shape[0], Size[0], Size[1], Size[2]])

	MaxAcc = 0
	MinLoss = float("Inf")
	MaxIndex = 0
	for i in range(len(AllCombos)):
		SuppData = support_index_to_data(TrainData, AllCombos[i])
		Acc, LossVal = Sess.run([Accuracy, Loss],
			feed_dict = {InputData: ValidationData, InputLabels: ValidationLabels, SupportData: SuppData, KeepProb: 1.0})
		if (LossVal < MinLoss):
			MaxAcc = Acc
			MinLoss = LossVal
			MaxIndex = i

	print("Best Index: {}".format(MaxIndex))
	print("Maximum Accuracy: {}".format(MaxAcc))
	print("Minimum Loss: {}".format(MinLoss))

	print("Testing best combination of supports...")
	SuppData = support_index_to_data(TrainData, AllCombos[MaxIndex], class_size = TestSize)
	TestData, TestLabels = get_test_data(datalist)
	TestData = np.reshape(TestData, [NumClasses * TestSize, Size[0], Size[1], Size[2]])

	#loop through different pieces of the TestData
	#stride = NumClasses * ValidationSize
	accuracy = 0
	count = 0
	for k in range(10):
		permutation = np.random.permutation(TestData.shape[0])
		TestData = TestData[permutation]
		TestLabels = TestLabels[permutation]

		Acc, c = Sess.run([Accuracy, Correct],
			feed_dict = {InputData: TestData, InputLabels: TestLabels, SupportData: SuppData, KeepProb: 1.0})
		count += 1
		accuracy += Acc

	accuracy /= count
	print("Independent Test Accuracy (best supports):", accuracy)

	#print('Saving model...')
	#print(Saver.save(Sess, "./saved/model"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
