#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import datetime
from scipy import misc
import os
import cv2

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
NumIteration = 200000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 5 # number of output classes
NumSupportsPerClass = 2
NumClassesInSubSet = 5
TrainSize = 10
TestSize = 10
EvalFreq = 10 # evaluate on every 1000th iteration

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]]) # network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2]])
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
#KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

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
def get_train_data(datalist, train_size = TrainSize, test_size = TestSize, num_classes = NumClasses, Size = [28, 28]):

	class_nums = random.sample(range(0, len(datalist)), num_classes)
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

	train_labels = np.asarray([idx / train_size for idx in range(train_size * num_classes)])

	permutation = np.random.permutation(train_labels.shape[0])
	train_labels = train_labels[permutation]
	train_data = train_data[permutation]

	return train_data, train_labels

def get_test_data(datalist, train_size = TrainSize, test_size = TestSize, num_classes = NumClasses, Size = [28, 28], class_nums = None):

	if class_nums == None:
		class_nums = random.sample(range(0, len(datalist)), num_classes)

	dir_names = datalist[class_nums]

	images = []

	for dir_name in dir_names:
		images.append(['{}{}'.format(dir_name, img) for img in os.listdir(dir_name) if not img.startswith('.')])

	images = np.asarray(images)

	test_set = images[:, train_size : train_size + test_size]
	test_set = np.reshape(test_set, num_classes * test_size)

	test_data = np.zeros([test_size * num_classes, Size[0], Size[1]])

	for k in range(test_size * num_classes):
		img = misc.imread(test_set[k])
		test_data[k, :, :] = misc.imresize(img, (Size[0], Size[1]))

	test_labels = np.asarray([idx / test_size for idx in range(test_size * num_classes)])

	permutation = np.random.permutation(test_labels.shape[0])
	test_labels = test_labels[permutation]
	test_data = test_data[permutation]

	return test_data, test_labels

#make the list of extensions to be used by the get data functions
data_location = '../data'
datalist, _ = make_dir_list(data_location)

np.random.shuffle(datalist)
datalist = datalist[0 : NumClassesInSubSet]

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
			W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel]) # this should be 3 and 3 if it is CNN friendly
			Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.1))

			CurrentFilters = NumKernel
			ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'VALID') #VALID, SAME
			ConvResult= tf.add(ConvResult, Bias)

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
		Correct = tf.argmax(OneHotLabels, 1)
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
	for Step in range(1, NumIteration):

			TrainData, TrainLabels = get_train_data(datalist)
			QueryData, SupportDataList, Label = make_support_set(TrainData, TrainLabels)

			# execute teh session
			Summary, _, Acc, L, p, c, cp = Sess.run([SummaryOp, Optimizer, Accuracy, Loss, Pred, Correct, CorrectPredictions], feed_dict = {InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})

			if (Step % 10 == 0):
				print("Iteration: " + str(Step))
				print("Accuracy: " + str(Acc))
				print("Loss: " + str(L))

			# independent test accuracy
			if not Step % EvalFreq:
				TotalAcc = 0
				for i in range(BatchLength):
					TestData, TestLabels = get_test_data(datalist)
					TestData, SuppData, TestLabels = make_support_set(TestData, TestLabels)

					Acc = Sess.run(Accuracy, feed_dict = {InputData: TestData, InputLabels: TestLabels, SupportData: SuppData})
					TotalAcc += Acc

				TotalAcc /= i
				print("Independent Test set: ", TotalAcc)
			SummaryWriter.add_summary(Summary, Step)

	print('Saving model...')
	print(Saver.save(Sess, "./saved/model"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)