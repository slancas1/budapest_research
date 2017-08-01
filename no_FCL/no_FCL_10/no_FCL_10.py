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
NumIteration = 250000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 10 # number of output classes
Dropout = 0.5 # droupout parameters in the FNN layer - currently not used
EvalFreq = 1000 # evaluate on every 100th iteration

# load data
path = '../data'
TrainData = np.load('{}/full_train_images.npy'.format(path))
TrainLabels = np.load('{}/full_train_labels.npy'.format(path))
TestData = np.load('{}/full_test_images.npy'.format(path))
TestLabels = np.load('{}/full_test_labels.npy'.format(path))

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [BatchLength, Size[0], Size[1], Size[2]]) # network input
InputLabels = tf.placeholder(tf.int32, [BatchLength]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
#OneHotLabels = (OneHotLabels * 2) - 1
KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

NumKernels = [32, 32, 32, 32, 10]
def MakeConvNet(Input, Size):
	CurrentInput = Input
	CurrentFilters = Size[2] # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(5): # number of layers
		with tf.variable_scope('conv' + str(i)):
				NumKernel = NumKernels[i]
				#W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel])
				W = tf.Variable(tf.random_normal([3, 3, CurrentFilters, NumKernel], stddev = 0.1), name = "W")
				#Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.0))

				CurrentFilters = NumKernel
				ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'SAME') #VALID, SAME
				#ConvResult= tf.add(ConvResult, Bias)

				#ReLU = tf.nn.relu(ConvResult)

				# bounded leaky ReLU
				alpha = 0.01
				ReLU = tf.maximum(-1 + alpha * (ConvResult + 1), ConvResult)
				ReLU = tf.maximum(1 + alpha * (ReLU - 1), ReLU)

				CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')

	#Out = tf.nn.dropout(CurrentInput, KeepProb)
	#return Out
	return CurrentInput

# construct model
OutMaps = MakeConvNet(InputData, Size)
OutShape = OutMaps.shape

# define loss and optimizer
# want to try and minimize the loss
with tf.name_scope('loss'):
		#LabelIndices = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.argmax(OneHotLabels, 1), 1), 1), 1)
		LabelIndices = tf.expand_dims(tf.expand_dims(OneHotLabels, 1), 1)
		GTMap = tf.tile(LabelIndices, tf.stack([1, OutShape[1], OutShape[2], 1])) * 2 - 1
		#GTMap = tf.tile(LabelIndices, tf.stack([1, OutShape[1], OutShape[2], 1]))
		#GTMap = tf.tile(LabelIndices, tf.stack([1, OutShape[1], OutShape[2], OutShape[3]]))
		GTMap = tf.cast(GTMap, tf.float32)
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

		#DiffZeros = tf.reduce_mean(tf.subtract(Zeros, OutMaps), [1, 2])
		DiffNegOnes = tf.square(tf.reduce_mean(tf.subtract(NegOnes, OutMaps), [1, 2]))
		DiffOne = tf.square(tf.reduce_mean(tf.subtract(Ones, OutMaps), [1, 2]))

		DiffList = []
		for i in range(NumClasses):
			#DiffList.append(tf.square(tf.reduce_sum(DiffZeros, 1) - DiffZeros[:, i] + DiffOne[:, i]))
			DiffList.append(tf.reduce_sum(DiffNegOnes, 1) - DiffNegOnes[:, i] + DiffOne[:, i])

		Diffs = tf.stack(DiffList)

		predictions = tf.argmin(Diffs, 0)
		correctPredictions = tf.equal(tf.cast(predictions, tf.int32), InputLabels)
		Accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

# initializing the variables
Init = tf.global_variables_initializer()

# create sumamries, these will be shown on tensorboard
#-------------------------------------------------------

# histogram summaries about the distribution of the variables
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
		#Summary, _, Acc, L = Sess.run([SummaryOp, Optimizer, Accuracy, Loss], feed_dict = {InputData: Data, InputLabels: Label, KeepProb: Dropout})
		Summary, _, Acc, L = Sess.run([SummaryOp, Optimizer, Accuracy, Loss], feed_dict = {InputData: Data, InputLabels: Label})

		if (Step % 100 == 0):
			print("Iteration: " + str(Step))
			print("Accuracy: " + str(Acc))
			print("Loss: " + str(L))

		# independent test accuracy
		if (Step % EvalFreq == 0):
			TotalAcc = 0
			Data = np.zeros([BatchLength] + Size)
			for i in range(0, TestData.shape[0], BatchLength): # third parameter is the stride length
				if TestData.shape[0] - i < 25:
					break
				Data = TestData[i : (i + BatchLength)]
				Label = TestLabels[i : (i + BatchLength)]
				#response = Sess.run(predictions, feed_dict = {InputData: Data, KeepProb: 1.0})
				response = Sess.run(predictions, feed_dict = {InputData: Data})
				for i in range(len(response)):
					if response[i] == Label[i]:
						TotalAcc += 1
			print("Independent Test set: " + str(float(TotalAcc) / TestData.shape[0]))
		SummaryWriter.add_summary(Summary, Step)
		Step += 1

	print('Saving model...')
	print(Saver.save(Sess, "./saved/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
