#!/usr/bin/env python2.7

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

# set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
#print dt
flags.DEFINE_string('summary_dir', '/tmp/tutorial/{}'.format(dt), 'Summaries directory')

# parameters
BatchLength = 32  # 32 images are in a minibatch
#Size = [105, 105, 1] # input img will be resized to this size
Size = [28, 28, 1]
NumIteration = 15000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
NumSupportsPerClass = 2
NumClassesInSubSet = 2
TrainSize = 7
ValidationSize = 4
TestSize = 9
EvalFreq = 200 # evaluate on every 1000th iteration

if (TrainSize + ValidationSize + TestSize != 20):
	print("DATA NOT PROPERLY SPLIT")
	exit(1)

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]]) # network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2]])
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
#KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

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
data_location = '../data'
datalist, _ = make_dir_list(data_location)

#np.random.shuffle(datalist)
datalist = datalist[0 : NumClassesInSubSet]

def all_support_combos(Data, Labels):
	SupportDataList = []
	comb = ncr(TrainSize, NumSupportsPerClass)

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
	SupportDataList = []
	QueryDataList = []
	QueryLabelList = []

	for i in range(BatchLength):

		QueryClass = np.random.randint(NumClasses)
		QueryIndices = np.argwhere(Labels == QueryClass)
		QueryIndex = QueryIndices[0]

		QueryDataList.append(Data[QueryIndex])
		QueryLabelList.append(Labels[QueryIndex])

		SupportDataList.append([])

		for j in range(NumClasses):
			if (j == QueryClass):
				SupportDataList[i].append(np.squeeze(Data[QueryIndices[1 : 1 + NumSupportsPerClass]], 1))
			else:
				SupportIndices = np.argwhere(Labels == j)
				SupportDataList[i].append(np.squeeze(Data[SupportIndices[0 : NumSupportsPerClass]], 1))


	QueryData = np.reshape(QueryDataList, [BatchLength, Size[0], Size[1], Size[2]])
	SupportDataList = np.reshape(SupportDataList, [BatchLength, NumClasses, NumSupportsPerClass, Size[0], Size[1], Size[2]])
	SupportDataList = np.transpose(SupportDataList, (0, 2, 1, 3, 4, 5))
	Label = np.reshape(QueryLabelList, [BatchLength])
	return QueryData, SupportDataList, Label

NumKernels = [32, 32, 32]
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

	return CurrentInput

with tf.name_scope('network'):

	encodedQuery = MakeConvNet(InputData, Size, First = True)
	SupportList = []
	QueryList = []

	for i in range(NumClasses):
		for k in range(NumSupportsPerClass):
			SupportList.append(MakeConvNet(SupportData[:, k, i, :, :, :], Size))
			QueryList.append(encodedQuery)

	QueryRepeat = tf.stack(QueryList)
	Supports = tf.stack(SupportList)


# define loss and optimizer
with tf.name_scope('loss'):
	'''calculate cosine similarity between encodedQuery and everything in Supports
	(A*B)/(|A||B|)'''

	DotProduct = tf.reduce_sum(tf.multiply(QueryRepeat, Supports), [2, 3, 4])
	#QueryMag = tf.sqrt(tf.reduce_sum(tf.square(QueryRepeat), [2, 3, 4]))
	SupportsMag = tf.sqrt(tf.reduce_sum(tf.square(Supports), [2, 3, 4]))
	CosSim = DotProduct / tf.clip_by_value(SupportsMag, 1e-10, float("inf"))

	CosSim = tf.reshape(CosSim, [NumClasses, NumSupportsPerClass, -1])
	CosSim = tf.transpose(tf.reduce_sum(CosSim, 1))

	probs = tf.nn.softmax(CosSim)

	Loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(OneHotLabels, CosSim))

with tf.name_scope('optimizer'):
		# use ADAM optimizer this is currently the best performing training algorithm in most cases
		Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
		#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
		Pred = tf.argmax(probs, 1)
		#Correct = tf.argmax(OneHotLabels, 1)
		Correct = tf.cast(InputLabels, tf.int64)
		CorrectPredictions = tf.equal(Pred, Correct)
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

	# keep training until reach max iterations - other stopping criterion could be added
	for Step in range(1, NumIteration + 1):

			TrainData, TrainLabels = get_train_data(datalist)
			# need to change make_support_set to create all combinations
			# need to calculate the accuracy for each combination and keep track of which pair of supports produces the max accuracy
			QueryData, SupportDataList, Label = make_support_set(TrainData, TrainLabels)

			# execute teh session
			Summary, _, Acc, L, p, c, cp = Sess.run([SummaryOp, Optimizer, Accuracy, Loss, Pred, Correct, CorrectPredictions],
				feed_dict = {InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})

			if (Step % 50 == 0):
				print("Iteration: " + str(Step))
				print("Accuracy: " + str(Acc))
				print("Loss: " + str(L))

			# independent test accuracy
			if not Step % EvalFreq:
				TotalAcc = 0
				count = 0
				for i in range(BatchLength):
					TestData, TestLabels = get_test_data(datalist)
					TestData, SuppData, TestLabels = make_support_set(TestData, TestLabels)

					Acc = Sess.run(Accuracy, feed_dict = {InputData: TestData, InputLabels: TestLabels, SupportData: SuppData})
					TotalAcc += Acc
					count += 1
				TotalAcc = TotalAcc / count
				print("Independent Test set: ", TotalAcc)
			SummaryWriter.add_summary(Summary, Step)

	print("Finding best supports...")

	AllCombos = all_support_combos(TrainData, TrainLabels)
	ValidationData, ValidationLabels = get_validation_data(datalist)
	ValidationData = np.reshape(ValidationData, [ValidationData.shape[0], Size[0], Size[1], Size[2]])

	MinLoss = float("Inf")
	MinIndex = 0
	for i in range(len(AllCombos)):
		SuppData = support_index_to_data(TrainData, AllCombos[i])
		LossVal = Sess.run([Accuracy], feed_dict = {InputData: ValidationData, InputLabels: ValidationLabels, SupportData: SuppData})
		if (LossVal < MinLoss):
			MinLoss = LossVal
			MinIndex = i

	SuppData = support_index_to_data(TrainData, AllCombos[MinIndex], class_size = TestSize)

	print("Testing best combination of supports...")
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

		Acc, p, c = Sess.run([Accuracy, Pred, Correct], feed_dict = {InputData: TestData, InputLabels: TestLabels, SupportData: SuppData})
		count += 1
		accuracy += Acc
		print(Acc)

	accuracy /= count
	print("Independent Test Accuracy (best supports):", accuracy)

	print('Saving model...')
	print(Saver.save(Sess, "./saved/model"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
