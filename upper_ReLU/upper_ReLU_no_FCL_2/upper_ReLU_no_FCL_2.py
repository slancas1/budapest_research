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
BatchLength = 25  # 25 images are in a minibatch
Size = [28, 28, 1] # input img will be resized to this size
NumIteration = 100000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
Dropout = 0.8 # droupout parameters in the FNN layer - currently not used
EvalFreq = 10 # evaluate on every 100th iteration

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
InputData = tf.placeholder(tf.float32, [BatchLength, Size[0], Size[1], Size[2]]) # network input
InputLabels = tf.placeholder(tf.int32, [BatchLength]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

NumKernels = [4, 4, 4, 1]
def MakeConvNet(Input, Size):
	CurrentInput = Input
	CurrentFilters = Size[2] # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(4): # number of layers
		with tf.variable_scope('conv' + str(i)):
				NumKernel = NumKernels[i]
				#W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel])
				W = tf.Variable(tf.random_normal([3, 3, CurrentFilters, NumKernel], stddev = 0.1), name = "W")
				#Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.0))

				CurrentFilters = NumKernel
				ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'SAME') #VALID, SAME
				#ConvResult= tf.add(ConvResult, Bias)

				# add batch normalization
				'''beta = tf.get_variable('beta', [NumKernel], initializer = tf.constant_initializer(0.0))
				gamma = tf.get_variable('gamma', [NumKernel], initializer = tf.constant_initializer(1.0))
				Mean, Variance = tf.nn.moments(ConvResult, [0, 1, 2])
				PostNormalized = tf.nn.batch_normalization(ConvResult, Mean, Variance, beta, gamma, 1e-10)'''

				#ReLU = tf.nn.relu(ConvResult)

				# leaky ReLU
				alpha = 0.01
				#ReLU = tf.maximum(alpha * ConvResult, ConvResult)
				ReLU = tf.maximum((-1 + alpha * (ConvResult + 1)), ConvResult)
				ReLU = tf.minimum((1 + alpha * (ReLU - 1)), ReLU) # do the -1 to the ReLU to shift the data

				CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')

	return CurrentInput
	'''# add fully connected network
	with tf.variable_scope('FC'):
		CurrentShape = CurrentInput.get_shape()
		FeatureLength = int(CurrentShape[1] * CurrentShape[2] * CurrentShape[3])
		FC = tf.reshape(CurrentInput, [-1, FeatureLength])
		W = tf.get_variable('W', [FeatureLength, NumClasses])
		FC = tf.matmul(FC, W)
		Bias = tf.get_variable('Bias', [NumClasses])
		FC = tf.add(FC, Bias)
		Out = tf.nn.dropout(FC, KeepProb)
	return Out'''

# construct model
OutMaps = MakeConvNet(InputData, Size)
OutShape = OutMaps.shape
OneMap = tf.ones([BatchLength, OutShape[1], OutShape[2], OutShape[3]], tf.float32)

# define loss and optimizer
# want to try and minimize the loss
with tf.name_scope('loss'):
		LabelIndices = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.argmax(OneHotLabels, 1), 1), 1), 1)
		GTMap = tf.tile(LabelIndices, tf.stack([1, OutShape[1], OutShape[2], OutShape[3]]))
		GTMap = tf.cast(GTMap, tf.float32)
		GTMap = (GTMap * 2) - 1
		DiffMap = tf.square(tf.subtract(GTMap, OutMaps))
		Loss = tf.reduce_sum(DiffMap)

# this is where the training begins and the training uses the loss -- want to try and minimize it
with tf.name_scope('optimizer'):
		# use ADAM optimizer this is currently the best performing training algorithm in most cases
		Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
		#Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
		#Zeros = tf.zeros(OutShape, tf.float32)
		NegOnes = tf.ones(OutShape, tf.float32) * -1
		Ones = tf.ones(OutShape, tf.float32)

		#sqdiff0 = tf.square(tf.subtract(Zeros, OutMaps))
		sqdiffneg1 = tf.square(tf.subtract(NegOnes, OutMaps))
		sqdiff1 = tf.square(tf.subtract(Ones, OutMaps))

		#rdmdiff0 = tf.reduce_mean(sqdiff0, [1, 2, 3])
		rdmdiffneg1 = tf.reduce_mean(sqdiffneg1, [1, 2, 3])
		rdmdiff1 = tf.reduce_mean(sqdiff1, [1, 2, 3])

		predictions = tf.argmin([rdmdiffneg1, rdmdiff1], 0)
		correctPredictions = tf.equal(tf.cast(predictions, tf.int32), InputLabels)

		Accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

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

		# execute the session
		Summary, _, Acc, L = Sess.run([SummaryOp, Optimizer, Accuracy, Loss], feed_dict = {InputData: Data, InputLabels: Label})

		# independent test accuracy
		if (Step % EvalFreq == 0):
			TotalAcc = 0
			Data = np.zeros([BatchLength] + Size)
			for i in range(0, TestData.shape[0], BatchLength): # third parameter is the stride length
				if TestData.shape[0] - i < 25:
					break
				Data = TestData[i : (i + BatchLength)]
				Label = TestLabels[i : (i + BatchLength)]
				response = Sess.run(predictions, feed_dict = {InputData: Data})
				for i in range(len(response)):
					if response[i] == Label[i]:
						TotalAcc += 1
			print("Iteration: " + str(Step))
			print("Accuracy: " + str(Acc))
			print("Loss: " + str(L))
			print("Independent Test set: " + str(float(TotalAcc) / TestData.shape[0]))
		SummaryWriter.add_summary(Summary, Step)
		Step += 1

	print('Saving model...')
	print(Saver.save(Sess, "./saved/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
