#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import datetime
#from tensorflow.python.client import timeline

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
BatchLength = 50  # 32 images are in a minibatch
Size = [28, 28, 1] # input img will be resized to this size
NumIteration = 100
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 10 # number of output classes
Dropout = 0.5 # droupout parameters in the FNN layer - currently not used
EvalFreq = 50 # evaluate on every 1000th iteration

# load data
TrainData = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/python-mnist/full_train_images.npy')
TrainLabels = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/python-mnist/full_train_labels.npy')
TestData = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/python-mnist/full_test_images.npy')
TestLabels = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/python-mnist/full_test_labels.npy')

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]]) # network input
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

NumKernels = [32, 64]
def MakeConvNet(Input, Size):
	CurrentInput = Input
	CurrentFilters = Size[2] # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(2): # number of layers
		with tf.variable_scope('conv' + str(i)):
				NumKernel = NumKernels[i]
				W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel]) # this should be 3 and 3 if it is CNN friendly
				Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.1))

				CurrentFilters = NumKernel
				ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'SAME') #VALID, SAME
				ConvResult= tf.add(ConvResult, Bias)

				# add batch normalization
				'''beta = tf.get_variable('beta', [NumKernel], initializer = tf.constant_initializer(0.0))
				gamma = tf.get_variable('gamma', [NumKernel], initializer = tf.constant_initializer(1.0))
				Mean, Variance = tf.nn.moments(ConvResult, [0, 1, 2])
				PostNormalized = tf.nn.batch_normalization(ConvResult, Mean, Variance, beta, gamma, 1e-10)'''

				# ReLU = tf.nn.relu(ConvResult)

				# leaky ReLU
				alpha = 0.01
				ReLU = tf.maximum(alpha * ConvResult, ConvResult)

				CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME') # this should be 1, 1, 1, 1 for both if the network is CNN friendly

	# add fully connected network
	with tf.variable_scope('FC'):
		CurrentShape = CurrentInput.get_shape()
		FeatureLength = int(CurrentShape[1] * CurrentShape[2] * CurrentShape[3])
		FC = tf.reshape(CurrentInput, [-1, FeatureLength])
		W = tf.get_variable('W', [FeatureLength, NumClasses])
		FC = tf.matmul(FC, W)
		Bias = tf.get_variable('Bias', [NumClasses])
		FC = tf.add(FC, Bias)
		Out = tf.nn.dropout(FC, KeepProb)
		alpha = 0.01
		ReLU = tf.maximum(alpha * FC, FC)
	return Out

# construct model
PredWeights = MakeConvNet(InputData, Size)

# define loss and optimizer
with tf.name_scope('loss'):
		Loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(OneHotLabels, PredWeights))

with tf.name_scope('optimizer'):
		# use ADAM optimizer this is currently the best performing training algorithm in most cases
		Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
		#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
		CorrectPredictions = tf.equal(tf.argmax(PredWeights, 1), tf.argmax(OneHotLabels, 1))
		Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))

# initializing the variables
Init = tf.global_variables_initializer()

# create sumamries, these will be shown on tensorboard

# histogram sumamries about the distributio nof the variables
for v in tf.trainable_variables():
	tf.summary.histogram(v.name[:-2], v)

# create image summary from the first 10 images
tf.summary.image('images', TrainData[1 : 10, :, :, :], max_outputs = 50)

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
	runOptions = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
	runMetaData = tf.RunMetadata()

	# keep training until reach max iterations - other stopping criterion could be added
	for Step in range(1, NumIteration + 1):

		# create train batch - select random elements for training
		TrainIndices = random.sample(range(TrainData.shape[0]), BatchLength)
		Data = TrainData[TrainIndices, :, :, :]
		Label = TrainLabels[TrainIndices]
		Label = np.reshape(Label, (BatchLength))

		# execute teh session
		Summary, _, Acc, L = Sess.run([SummaryOp, Optimizer, Accuracy, Loss],
			options = runOptions, run_metadata = runMetaData,
			feed_dict = {InputData: Data, InputLabels: Label, KeepProb: Dropout})

		SummaryWriter.add_run_metadata(runMetaData, 'step%d' % Step)
		SummaryWriter.add_summary(Summary, Step)

		'''tl = timeline.Timeline(runMetaData.step_stats)
		ctf = tl.generate_chrome_trace_format()
		with open('timeline_train.json', 'w') as f:
			f.write(ctf)'''

		if (Step % 50 == 0):
			print("Iteration: " + str(Step))
			print("Accuracy: " + str(Acc))
			print("Loss: " + str(L))

		# independent test accuracy
		if (Step % EvalFreq == 0) or Step == NumIteration - 1:
			TotalAcc = 0;
			Data = np.zeros([1] + Size)
			for i in range(0, TestData.shape[0]):
				Data[0] = TestData[i]
				Label = TestLabels[i]
				response = Sess.run(PredWeights,
					#options = runOptions, run_metadata = runMetaData,
					feed_dict = {InputData: Data, KeepProb: 1.0})

				'''tl = timeline.Timeline(runMetaData.step_stats)
				ctf = tl.generate_chrome_trace_format()
				with open('timeline_test.json', 'w') as f:
					f.write(ctf)'''

				if np.argmax(response) == Label:
					TotalAcc += 1
			print("Independent Test set: " + str(float(TotalAcc) / TestData.shape[0]))
		SummaryWriter.add_summary(Summary, Step)
		Step += 1

	print('Saving model...')
	print(Saver.save(Sess, "./saved/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
