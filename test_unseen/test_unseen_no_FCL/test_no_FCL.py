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
BatchLength = 25 # 25 images are in a minibatch
Size = [28, 28, 1] # input img will be resized to this size
NumIteration = 100000
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
Dropout = 0.8 # droupout parameters in the FNN layer - currently not used
EvalFreq = 100 # evaluate on every 100th iteration

# load data
path = '../../data'
TrainData = np.load('{}/6and9_train_images.npy'.format(path))
TrainLabels = np.load('{}/6and9_train_labels.npy'.format(path))
TestData = np.load('{}/full_test_images.npy'.format(path))
TestLabels = np.load('{}/full_test_labels.npy'.format(path))
#TestLabels = TestLabels[:-2]

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

				ReLU = tf.nn.relu(ConvResult)

				# leaky ReLU
				#alpha = 0.01
				#ReLU = tf.maximum(alpha * ConvResult, ConvResult)

				CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME')

	return CurrentInput

# construct model
OutMaps = MakeConvNet(InputData, Size)
OutShape = OutMaps.shape
OneMap = tf.ones([BatchLength, OutShape[1], OutShape[2], OutShape[3]], tf.float32)

with tf.name_scope('accuracy'):
		OutMaps = OutMaps - 1
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
#tf.summary.scalar("loss", Loss)
tf.summary.scalar("accuracy", Accuracy)

SummaryOp = tf.summary.merge_all()

# limits the amount of GPU you can use so you don't tie up the server
conf = tf.ConfigProto(allow_soft_placement = True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.2

checkpoint = './saved/model'

# launch the session with default graph
with tf.Session(config = conf) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir, tf.get_default_graph())
	Saver = tf.train.Saver()
	Saver.restore(Sess, checkpoint)

	f = open("testdata.csv", "w")

	# keep training until reach max iterations - other stopping criterion could be added
	for k in range(0, len(TestLabels), BatchLength):

		# create train batch - select random elements for training
		#TrainIndices = Step
		Data = TestData[k : k + BatchLength, :, :, :]
		Label = TestLabels[k:k+BatchLength]
		Label = np.reshape(Label, (BatchLength))

		# execute the session
		Summary, Acc, RDMN1, RDM1, P = Sess.run([SummaryOp, Accuracy, rdmdiffneg1, rdmdiff1, predictions], feed_dict = {InputData: Data, InputLabels: Label})

		for i in range(BatchLength):
			f.write(str(Label[i]) + ", " + str(RDMN1[i]) + ", " + str(RDM1[i]) + ", "+ str(P[i]) + "\n")

		#SummaryWriter.add_summary(Summary, Step)

	f.close()

	#print('Saving model...')
	#print(Saver.save(Sess, "./saved/"))

#print("Optimization Finished!")
#print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
