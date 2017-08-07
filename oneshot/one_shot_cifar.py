#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import datetime
import os
from scipy import misc
import cv2

# set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
#print dt
flags.DEFINE_string('summary_dir', '/tmp/tutorial/{}'.format(dt), 'Summaries directory')

# if summary directory exist, delete the previous summaries
'''if tf.gfile.Exists(FLAGS.summary_dir):
	tf.gfile.DeleteRecursively(FLAGS.summary_dir)
	tf.gfile.MakeDirs(FLAGS.summary_dir)'''

# parameters
BatchLength = 32  # 32 images are in a minibatch
Size = [32, 32, 3] # input img will be resized to this size
NumIteration = 15000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
NumSupportsPerClass = 2
NumSupports = NumClasses * NumSupportsPerClass
#Dropout = 0.5 # droupout parameters in the FNN layer - currently not used
TrainSize = 10
TestSize = 200
EvalFreq = 200 # evaluate on every 1000th iteration

# load data
path = '../data'
TrainData = np.load('{}/Cifar_train_data.npy'.format(path))
TrainLabels = np.load('{}/Cifar_train_labels.npy'.format(path))
TestData = np.load('{}/Cifar_test_data.npy'.format(path))
TestLabels = np.load('{}/Cifar_test_labels.npy'.format(path))

ClassTypes = [6, 9]

TrainDataList = []
TrainLabelList = []

#randomize order
permutation = np.random.permutation(TrainData.shape[0])
TrainData = TrainData[permutation]
TrainLabels = TrainLabels[permutation]
for classnum in range(NumClasses):
	train_indices = np.argwhere(TrainLabels == ClassTypes[classnum])[:, 0]
	TrainDataList.append(TrainData[train_indices[0 : TrainSize]])
	TrainLabelList.append([classnum] * TrainSize)

TrainData = np.reshape(TrainDataList, [TrainSize * NumClasses, Size[0], Size[1], Size[2]])
TrainLabels = np.reshape(TrainLabelList, [TrainSize * NumClasses])

TestDataList = []
TestLabelList = []

#randomize order
permutation = np.random.permutation(TestData.shape[0])
TestData = TestData[permutation]
TestLabels = TestLabels[permutation]
for classnum in range(NumClasses):
	test_indices = np.argwhere(TestLabels == ClassTypes[classnum])[:, 0]
	TestDataList.append(TestData[test_indices[0 : TestSize]])
	TestLabelList.append([classnum] * TestSize)

TestData = np.reshape(TestDataList, [TestSize * NumClasses, Size[0], Size[1], Size[2]])
TestLabels = np.reshape(TestLabelList, [TestSize * NumClasses])

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]]) # network input
SupportData = tf.placeholder(tf.float32, [None, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2]])
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
#KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

def make_support_set(Data, Labels):
	SupportDataList = np.zeros((BatchLength, NumSupportsPerClass, NumClasses, Size[0], Size[1], Size[2]))
	QueryData = np.zeros((BatchLength, Size[0], Size[1], Size[2]))
	QueryLabel = np.zeros(BatchLength)

	for i in range(BatchLength):

		QueryClass = np.random.randint(NumClasses)
		QueryIndices = np.argwhere(Labels == QueryClass)
		permutation = np.random.permutation(QueryIndices.shape[0])
		QueryIndices = QueryIndices[permutation]
		QueryIndex = QueryIndices[0]


		QueryData[i, :, :, :] = np.reshape(Data[QueryIndex], (Size[0], Size[1], Size[2]))
		QueryLabel[i] = Labels[QueryIndex]

		for j in range(NumClasses):
			if (j == QueryClass):
				for k in range(NumSupportsPerClass):
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
			W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel]) # this should be 3 and 3 if it is CNN friendly
			Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.1))

			CurrentFilters = NumKernel
			ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'VALID') #VALID, SAME
			ConvResult = tf.add(ConvResult, Bias)

			# add batch normalization
			'''beta = tf.get_variable('beta', [NumKernel], initializer = tf.constant_initializer(0.0))
			gamma = tf.get_variable('gamma', [NumKernel], initializer = tf.constant_initializer(1.0))
			Mean, Variance = tf.nn.moments(ConvResult, [0, 1, 2])
			PostNormalized = tf.nn.batch_normalization(ConvResult, Mean, Variance, beta, gamma, 1e-10)'''

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

	CosSim = tf.reshape(CosSim, (NumClasses, NumSupportsPerClass, -1))
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
conf.gpu_options.per_process_gpu_memory_fraction = 0.2

# launch the session with default graph
with tf.Session(config = conf) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir, tf.get_default_graph())
	Saver = tf.train.Saver()

	# keep training until reach max iterations - other stopping criterion could be added

	for Step in range(1, NumIteration + 1):
			QueryData, SupportDataList, Label = make_support_set(TrainData, TrainLabels)

			# execute teh session
			Summary, _, Acc, L = Sess.run([SummaryOp, Optimizer, Accuracy, Loss], feed_dict = {InputData: QueryData, InputLabels: Label, SupportData: SupportDataList})

			if (Step % 100 == 0):
				print("Iteration: " + str(Step))
				print("Accuracy: " + str(Acc))
				print("Loss: " + str(L))

			# independent test accuracy
			if not Step % EvalFreq:
				TotalAcc = 0
				count = 0
				for i in range(0, 2 * (TestSize / BatchLength)):
					Data, SuppData, Label = make_support_set(TestData, TestLabels)
					response = Sess.run(Accuracy, feed_dict = {InputData: Data, InputLabels: Label, SupportData: SuppData})
					TotalAcc += response
					count += 1.0
				TotalAcc /= count
				print("Independent Test set: ", TotalAcc)
			SummaryWriter.add_summary(Summary, Step)

	print('Saving model...')
	print(Saver.save(Sess, "./saved/model"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
