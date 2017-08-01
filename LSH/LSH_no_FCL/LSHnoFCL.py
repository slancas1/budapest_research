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
BatchLength = 14  # 25 images are in a minibatch`
Size = [28, 28, 1] # input img will be resized to this size
NumIteration = 100
LearningRate = 1e-4 # learning rate of the algorithm
NumClasses = 2 # number of output classes
Dropout = 0.5 # droupout parameters in the FNN layer - currently not used
EvalFreq = 1000 # evaluate on every 100th iteration

# load data
path = '../../data'
TrainData = np.load('{}/6and9_train_images.npy'.format(path))
TrainLabels = np.load('{}/6and9_train_labels.npy'.format(path))
TestData = np.load('{}/6and9_test_images.npy'.format(path))
TestLabels = np.load('{}/6and9_test_labels.npy'.format(path))

# create tensorflow graph
InputData = tf.placeholder(tf.float32, [BatchLength, Size[0], Size[1], Size[2]]) # network input
InputLabels = tf.placeholder(tf.int32, [BatchLength]) # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
KeepProb = tf.placeholder(tf.float32) # dropout (keep probability -currently not used)

NumKernels = [16, 16, 16]
def MakeConvNet(Input, Size):
	CurrentInput = Input
	CurrentFilters = Size[2] # the input dim at the first layer is 1, since the input image is grayscale
	for i in range(len(NumKernels)): # number of layers
		with tf.variable_scope('conv' + str(i)) as varscope:
				NumKernel = NumKernels[i]
				#W = tf.get_variable('W', [3, 3, CurrentFilters, NumKernel])
				W = tf.Variable(tf.random_normal([3, 3, CurrentFilters, NumKernel], stddev = 0.1), name = "W")
				#Bias = tf.get_variable('Bias', [NumKernel], initializer = tf.constant_initializer(0.0))

				CurrentFilters = NumKernel
				ConvResult = tf.nn.conv2d(CurrentInput, W, strides = [1, 1, 1, 1], padding = 'VALID') #VALID, SAME
				#ConvResult= tf.add(ConvResult, Bias)

				#ReLU = tf.nn.relu(ConvResult)

				# leaky ReLU
				alpha = 0.01
				ReLU = tf.maximum(alpha * ConvResult, ConvResult)
				# ksize = kernel size
				CurrentInput = tf.nn.max_pool(ReLU, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'VALID')

	CurrentInput = tf.nn.dropout(CurrentInput, KeepProb)

	return CurrentInput

# construct model
OutMaps = MakeConvNet(InputData, Size)
OutShape = OutMaps.shape
OutMaps = tf.reshape(OutMaps, [int(BatchLength), int(OutShape[1] * OutShape[2]), int(OutShape[3])])
#ValuesToHash = tf.reshape(InputData, [int(BatchLength), int(OutShape[1] * OutShape[2]), int(Size[2])])
#OneMap = tf.ones([BatchLength, OutShape[1], OutShape[2], OutShape[3]], tf.float32)


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

# this is where the training begins and the training uses the loss -- want to try and minimize it
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

		# execute the session
		Summary, _, Acc, L, P = Sess.run([SummaryOp, Optimizer, Accuracy, Loss, predictions],
			feed_dict = {InputData: Data, InputLabels: Label, KeepProb: 0.8})

		SummaryWriter.add_run_metadata(runMetaData, 'step%d' % Step)
		SummaryWriter.add_summary(Summary, Step)

		if (Step % 100 == 0):
			print("Iteration: " + str(Step))
			print("Accuracy: " + str(Acc))
			print("Loss: " + str(L))

		# independent test accuracy
		if not Step % EvalFreq:
			TotalAcc = 0
			count = 0
			Data = np.zeros([BatchLength] + Size)
			for i in range(BatchLength): # third parameter is the stride length
				if TestData.shape[0] - i < 25:
					break
				Data = TestData[i : (i + BatchLength)]
				Label = TestLabels[i : (i + BatchLength)]
				Acc, L = Sess.run([Accuracy, Loss], feed_dict = {InputData: Data, InputLabels: Label, KeepProb: 1.0})
				TotalAcc += Acc
				count += 1
			TotalAcc = TotalAcc / count
			print("Independent Test set: ", TotalAcc)
		SummaryWriter.add_summary(Summary, Step)

	print('Saving model...')
	print(Saver.save(Sess, "./saved/"))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir=" + FLAGS.summary_dir)
