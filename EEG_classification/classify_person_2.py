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
flags.DEFINE_string('summary_dir', '/tmp/EEG/{}'.format(dt), 'Summaries directory')

# parameters
BatchLength = 25  # 32 images are in a minibatch
Size = [2500, 1] # the number of data points per each iteration of a class
NumIteration = 1000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
NumSupportsPerClass = 2
#NumClassesInSubSet = 2
EvalFreq = 50 # evaluate on every 50th iteration


# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1]]) # network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1]])
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
#KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

# Load in EEG data
directory = './PeopleData/'
data_in = np.load('{}split_person_data.npy'.format(directory))
labels_in = np.load('{}split_person_labels.npy'.format(directory))

# restructure data to have a train and a test set
NumElementsPerClass = data_in.shape[0] / NumClasses
#print(NumElementsPerClass)
TrainSize = 15
TestSize = NumElementsPerClass - TrainSize
Data = np.zeros([NumClasses, NumElementsPerClass, Size[0]])
Labels = np.zeros([NumClasses, NumElementsPerClass])
for k in range(NumClasses):
	k_inds = np.argwhere(labels_in == k)[:, 0]
	Data[k] = data_in[k_inds]
	Labels[k] = labels_in[k_inds]
TrainData = Data[:, 0 : TrainSize]
TrainLabels = Labels[:, 0 : TrainSize]
TestData = Data[:, TrainSize : TrainSize + TestSize]
TestLabels = Labels[:, TrainSize : TrainSize + TestSize]

# reshape to put both classes in same dimension
TrainData = np.reshape(TrainData, [TrainSize * NumClasses, Size[0]])
TrainLabels = np.reshape(TrainLabels, [TrainSize * NumClasses])
TestData = np.reshape(TestData, [TestSize * NumClasses, Size[0]])
TestLabels = np.reshape(TestLabels, [TestSize * NumClasses])

# randomize order
permutation = np.random.permutation(TrainData.shape[0])
TrainData = TrainData[permutation]
TrainLabels = TrainLabels[permutation]
permutation = np.random.permutation(TestData.shape[0])
TestData = TestData[permutation]
TestLabels = TestLabels[permutation]

def make_support_set(Data, Labels):

	# randomize so you don't always pick the same data points
	permutation = np.random.permutation(Data.shape[0])
	Data = Data[permutation]
	Labels = Labels[permutation]

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
				SupportDataList[i].append(np.squeeze(Data[QueryIndices[1 : 1 + NumSupportsPerClass]], axis = 1))
			else:
				SupportIndices = np.argwhere(Labels == j)
				SupportDataList[i].append(np.squeeze(Data[SupportIndices[0 : NumSupportsPerClass]], axis = 1))

	QueryData = np.reshape(QueryDataList, [BatchLength, Size[0], Size[1]])
	SupportDataList = np.reshape(SupportDataList, [BatchLength, NumClasses, NumSupportsPerClass, Size[0], Size[1]])
	SupportDataList = np.transpose(SupportDataList, (0, 2, 1, 3, 4))
	Label = np.reshape(QueryLabelList, [BatchLength])
	return QueryData, SupportDataList, Label

NumKernels = [32, 32, 32]
def MakeConvNet(Input, Size, First = False):
	CurrentInput = Input
	CurrentInput = (CurrentInput / 255.0) - 0.5
	CurrentFilters = Size[-1] # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(len(NumKernels)): # number of layers
		with tf.variable_scope('conv' + str(i)) as varscope:
			if not First:
				varscope.reuse_variables()
			NumKernel = NumKernels[i]
			W = tf.get_variable('W',[7, CurrentFilters, NumKernel])

			Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.1))

			CurrentFilters = NumKernel

			ConvResult = tf.nn.conv1d(CurrentInput, W, stride = 5, padding = 'VALID') #VALID, SAME
			ConvResult= tf.add(ConvResult, Bias)

			# ReLU = tf.nn.relu(ConvResult)

			# leaky ReLU
			alpha = 0.01
			ReLU = tf.maximum(alpha * ConvResult, ConvResult)

			ReLU = tf.expand_dims(ReLU, 1)

			# this should be 1, 1, 1, 1 for both if the network is CNN friendly
			CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 1, 5, 1], strides = [1, 1, 1, 1], padding = 'VALID')
			CurrentInput = tf.squeeze(CurrentInput, squeeze_dims = 1)


	return CurrentInput

with tf.name_scope('network'):

	encodedQuery = MakeConvNet(InputData, Size, First = True)
	SupportList = []
	QueryList = []

	for i in range(NumClasses):
		for k in range(NumSupportsPerClass):
			SupportList.append(MakeConvNet(SupportData[:, k, i, :, :], Size))
			QueryList.append(encodedQuery)

	QueryRepeat = tf.stack(QueryList)
	Supports = tf.stack(SupportList)

# define loss and optimizer
with tf.name_scope('loss'):
	'''calculate cosine similarity between encodedQuery and everything in Supports
	(A*B)/(|A||B|)'''

	DotProduct = tf.reduce_sum(tf.multiply(QueryRepeat, Supports), [2, 3])
	#QueryMag = tf.sqrt(tf.reduce_sum(tf.square(QueryRepeat), [2, 3, 4]))
	SupportsMag = tf.sqrt(tf.reduce_sum(tf.square(Supports), [2, 3]))
	CosSim = DotProduct / tf.clip_by_value(SupportsMag, 1e-10, float("inf"))

	CosSim = tf.reshape(CosSim, [NumClasses, NumSupportsPerClass, -1])
	CosSim = tf.transpose(tf.reduce_sum(CosSim, 1))

	probs = tf.nn.softmax(CosSim)

	Loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(OneHotLabels, probs))

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
	for Step in range(1, NumIteration + 1):

			QueryData, SupportDataList, Label = make_support_set(TrainData, TrainLabels)

			# execute the session
			Summary, _, Acc, L, p, c, pbs = Sess.run([SummaryOp, Optimizer, Accuracy, Loss, Pred, Correct, probs],
				feed_dict = {InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})
			#print("Probabilities: {}".format(pbs[0:10]))
			#print("Predictions: {}".format(p[0:10]))
			#print("Correct Labels: {}".format(c[0:10]))

			if (Step % 10 == 0):
				print("Iteration: " + str(Step))
				print("Accuracy: " + str(Acc))
				print("Loss: " + str(L))

			# independent test accuracy
			if not Step % EvalFreq:
				TotalAcc = 0
				count = 0
				for i in range(BatchLength):
					TestDat, SuppData, TestLab = make_support_set(TestData, TestLabels)

					Acc = Sess.run(Accuracy, feed_dict = {InputData: TestDat, InputLabels: TestLab, SupportData: SuppData})
					TotalAcc += Acc
					count += 1
				TotalAcc = TotalAcc / count
				print("Independent Test set: ", TotalAcc)
			SummaryWriter.add_summary(Summary, Step)


	#print('Saving model...')
	#print(Saver.save(Sess, "./saved/model"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
