#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import datetime

# set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
#print dt
flags.DEFINE_string('summary_dir', '/tmp/tutorial/{}'.format(dt), 'Summaries directory')

# if summary directory exist, delete the previous summaries
if tf.gfile.Exists(FLAGS.summary_dir):
	tf.gfile.DeleteRecursively(FLAGS.summary_dir)
	tf.gfile.MakeDirs(FLAGS.summary_dir)

# parameters
BatchLength = 32  # 32 images are in a minibatch
Size = [28, 28, 1] # Input img will be resized to this size
NumIteration = 100000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
Dropout = 0.8 # droupout parameters in the FNN layer - currently not used
EvalFreq = 100 # evaluate on every 100th iteration

# load data
# load data
path = '../../data'
TrainData = np.load('{}/6and9_train_images.npy'.format(path))
TrainLabels = np.load('{}/6and9_train_labels.npy'.format(path))
TestData = np.load('{}/6and9_test_images.npy'.format(path))
TestLabels = np.load('{}/6and9_test_labels.npy'.format(path))

TrainLabels[TrainLabels == 6] = 0
TrainLabels[TrainLabels == 9] = 1
TestLabels[TestLabels == 6] = 0
TestLabels[TestLabels == 9] = 1

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]]) # network input
InputLabels = tf.placeholder(tf.int32, [None]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

NumKernels = [32, 32, 32, 1]
def MakeConvNet(Input, Size, KeepProb):
	CurrentInput = Input
	CurrentFilters = Size[2] # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(3): # number of layers
		with tf.variable_scope('conv' + str(i)):
				NumKernel = NumKernels[i]
				W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel])
				Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.0))

				CurrentFilters = NumKernel
				ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'VALID') #VALID, SAME
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

				CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'VALID')

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
	return Out

# construct model
PredWeights = MakeConvNet(InputData, Size, Dropout)

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

	Step = 1

	# keep training until reach max iterations - other stopping criterion could be added
	while Step < NumIteration:

		# create train batch - select random elements for training
		TrainIndices = random.sample(range(TrainData.shape[0]), BatchLength)
		Data = TrainData[TrainIndices, :, :, :]
		Label = TrainLabels[TrainIndices]
		Label = np.reshape(Label, (BatchLength))

		# execute teh session
		Summary, _, Acc, L = Sess.run([SummaryOp, Optimizer, Accuracy, Loss], feed_dict = {InputData: Data, InputLabels: Label})

		# independent test accuracy
		if (Step % EvalFreq == 0):
			TotalAcc = 0;
			Data = np.zeros([1] + Size)
			for i in range(0, TestData.shape[0]):
				Data[0] = TestData[i]
				Label = TestLabels[i]
				response = Sess.run(PredWeights, feed_dict = {InputData: Data})
				if np.argmax(response) == Label:
					TotalAcc += 1
			print("Iteration: " + str(Step))
			print("Accuracy: " + str(Acc))
			print("Loss: " + str(L))
			print("Independent Test set: " + str(float(TotalAcc) / TestData.shape[0]))
		SummaryWriter.add_summary(Summary, Step)
		Step += 1

	print('Saving model...')
	print(Saver.save(Sess, "./saved/model"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
