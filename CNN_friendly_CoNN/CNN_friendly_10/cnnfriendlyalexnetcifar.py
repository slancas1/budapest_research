#!/usr/bin/env python2.7

from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import random
import cv2

# set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
now = datetime.datetime.now()
dataset = 'mnist'
dt = ('%s_%s_%s_%s' % (now.month, now.day, now.hour, now.minute))
flags.DEFINE_string('summary_dir', '/tmp/cnn_no_fcl/{}/{}'.format(dataset, dt), 'Summaries directory')

# if summary directory exist, delete the previous summaries
# if tf.gfile.Exists(FLAGS.summary_dir):
#	 tf.gfile.DeleteRecursively(FLAGS.summary_dir)
#	 tf.gfile.MakeDirs(FLAGS.summary_dir)

# Parameters
BatchLength = 32  # 32 images are in a minibatch
#Size = [227, 227, 3]  # Input img will be resized to this size
#Size = [128, 128, 1]
Size = [128, 128, 3]
NumIteration = 25000
LearningRate = 1e-4  # learning rate of the algorithm
NumClasses = 10  # number of output classes
Dropout = 0.5  # droupout parameters in the FNN layer - currently not used
EvalFreq = 100  # evaluate on every 100th iteration


# load data
TrainData = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/CIFAR/Cifar_train_data.npy')
TrainLabels = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/CIFAR/Cifar_train_labels.npy')
TestData = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/CIFAR/Cifar_test_data.npy')
TestLabels = np.load('/Users/Sophie/Desktop/Google Drive/Budapest/Code/CIFAR/Cifar_test_labels.npy')

'''# load data
TrainData = np.load('/Users/Sophie/Desktop/School Stuff/Budapest/Code/python-mnist/full_train_images.npy')
TrainLabels = np.load('/Users/Sophie/Desktop/School Stuff/Budapest/Code/python-mnist/full_train_labels.npy')
TestData = np.load('/Users/Sophie/Desktop/School Stuff/Budapest/Code/python-mnist/full_test_images.npy')
TestLabels = np.load('/Users/Sophie/Desktop/School Stuff/Budapest/Code/python-mnist/full_test_labels.npy')'''

# Create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2]])  # network input
InputLabels = tf.placeholder(tf.int32, [None])  # desired network output
OneHotLabels = tf.one_hot(InputLabels, NumClasses)
KeepProb = tf.placeholder(tf.float32)  # dropout (keep probability)

def AddRelUfc(Input):
    return tf.nn.elu(Input)

def AddRelUconv(Input):
    return tf.nn.elu(Input)

def MakeAlexNet(Input, Size, KeepProb):
    CurrentInput = Input  # 227, 227, 3
    CurrentInput = CurrentInput / 255.0
    with tf.variable_scope('conv1'):
        # first convolution
        W = tf.get_variable('W', [11, 11, Size[2], 96])
        Bias = tf.get_variable(
            'Bias', [96], initializer = tf.constant_initializer(0.1))
        ConvResult1 = tf.nn.conv2d(CurrentInput, W, strides = [1, 4, 4, 1], padding = 'SAME')  # VALID, SAME
        ConvResult1 = tf.add(ConvResult1, Bias)
        # first relu
        ReLU1 = AddRelUconv(ConvResult1)
        # response normalization
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        Norm1 = tf.nn.local_response_normalization(ReLU1, depth_radius = radius, alpha = alpha, beta = beta, bias = bias)
        # first pooling
        Pool1 = tf.nn.max_pool(Norm1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    with tf.variable_scope('conv2'):
        # second convolution
        W = tf.get_variable('W', [5, 5, 96, 256])
        Bias = tf.get_variable('Bias', [256], initializer = tf.constant_initializer(0.1))
        ConvResult2 = tf.nn.conv2d(Pool1, W, strides = [1, 1, 1, 1], padding = 'SAME')  # VALID, SAME
        ConvResult2 = tf.add(ConvResult2, Bias)
        # second relu
        ReLU2 = AddRelUconv(ConvResult2)
        # response normalization
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        Norm2 = tf.nn.local_response_normalization(ReLU2, depth_radius = radius, alpha = alpha, beta = beta, bias = bias)
        # second pooling
        Pool2 = tf.nn.max_pool(Norm2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    with tf.variable_scope('conv3'):
        # third convolution
        W = tf.get_variable('W', [3, 3, 256, 384])
        Bias = tf.get_variable('Bias', [384], initializer = tf.constant_initializer(0.1))
        ConvResult3 = tf.nn.conv2d(Pool2, W, strides = [1, 1, 1, 1], padding = 'SAME')  # VALID, SAME
        ConvResult3 = tf.add(ConvResult3, Bias)
        # third relu
        ReLU3 = AddRelUconv(ConvResult3)
    with tf.variable_scope('conv4'):
        # fourth convolution
        W = tf.get_variable('W', [3, 3, 384, 384])
        Bias = tf.get_variable('Bias', [384], initializer = tf.constant_initializer(0.1))
        ConvResult4 = tf.nn.conv2d(ReLU3, W, strides = [1, 1, 1, 1], padding = 'SAME')  # VALID, SAME
        ConvResult4 = tf.add(ConvResult4, Bias)
        # fourth relu
        ReLU4 = AddRelUconv(ConvResult4)
    with tf.variable_scope('conv5'):
        # fifth convolution
        W = tf.get_variable('W', [3, 3, 384, 256])
        Bias = tf.get_variable('Bias', [256], initializer = tf.constant_initializer(0.1))
        ConvResult5 = tf.nn.conv2d(ReLU4, W, strides = [1, 1, 1, 1], padding = 'SAME')  # VALID, SAME
        ConvResult5 = tf.add(ConvResult5, Bias)
        # fifth relu
        ReLU5 = AddRelUconv(ConvResult5)
        # fifth pooling
        Pool3 = tf.nn.max_pool(ReLU5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    with tf.variable_scope('conv_out'):
        # fifth convolution
        W = tf.get_variable('W', [3, 3, 256, 10])
        Bias = tf.get_variable('Bias', [10], initializer = tf.constant_initializer(0.1))
        ConvResultOut = tf.nn.conv2d(Pool3, W, strides = [1, 1, 1, 1], padding = 'SAME')  # VALID, SAME
        ConvResultOut = tf.add(ConvResultOut, Bias)
        # final relu
        Out = AddRelUconv(ConvResultOut)

    return Out

# Construct model
OutMaps = MakeAlexNet(InputData, Size, KeepProb)
OutShape = OutMaps.shape
OutShape = [BatchLength, OutShape[1], OutShape[2], OutShape[3]]


# Define loss and optimizer
with tf.name_scope('loss'):
    LabelIndices = tf.expand_dims(tf.expand_dims(OneHotLabels, 1), 1)  #32

    GTMap = tf.tile(LabelIndices, tf.stack([1, OutShape[1], OutShape[2], 1]) ) * 2 - 1

    GTMap = tf.cast(GTMap, tf.float32)
    print(GTMap.shape, OutMaps.shape)
    DiffMap = tf.square(tf.subtract(GTMap, OutMaps))
    Loss = tf.reduce_sum(DiffMap)

with tf.name_scope('optimizer'):
    # Use ADAM optimizer this is currently the best performing training algorithm in most cases
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
    Zeros = tf.ones(OutShape, tf.float32) * -1 #actually -1s
    Ones = tf.ones(OutShape, tf.float32)

    print(Zeros.shape)

    DiffZeros = tf.reduce_mean(tf.square(tf.subtract(Zeros, OutMaps)), [1, 2])
    DiffOnes = tf.reduce_mean(tf.square(tf.subtract(Ones, OutMaps)), [1, 2])

    DiffList = []
    for k in range(NumClasses):
        x = DiffZeros[:, k]
        y = tf.reduce_sum(DiffZeros, 1)
        DiffList.append(tf.reduce_sum(DiffZeros, 1) - DiffZeros[:, k] + DiffOnes[:, k])


    Diffs = tf.stack(DiffList)
    Pred = tf.argmin(Diffs, 0)
    CorrectPredictions = tf.equal(tf.cast(Pred, tf.int32), InputLabels)
    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))


# Initializing the variables
Init = tf.global_variables_initializer()

# create sumamries, these will be shown on tensorboard

# histogram sumamries about the distribution of the variables
for v in tf.trainable_variables():
    tf.summary.histogram(v.name[:-2], v)

# create image summary from the first 10 images
tf.summary.image('images', TrainData[1 : 10, :, :, :],  max_outputs = 50)

# create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
tf.summary.scalar("accuracy", Accuracy)


SummaryOp = tf.summary.merge_all()


# Launch the session with default graph
conf = tf.ConfigProto(allow_soft_placement = True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.2  # fraction of GPU used

with tf.device('/gpu:0'):
    with tf.Session(config = conf) as Sess:
        Sess.run(Init)
        SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir, tf.get_default_graph())
        Saver = tf.train.Saver()

        Step = 1
        # Keep training until reach max iterations - other stopping criterion could be added
        while Step <= NumIteration:

            # create train batch - select random elements for training
            TrainIndices = random.sample(range(TrainData.shape[0]), BatchLength)
            Data = TrainData[TrainIndices, :, :, :]
            InData = np.zeros((BatchLength, Size[0], Size[1], Size[2]))
            Label = TrainLabels[TrainIndices]
            Label = np.reshape(Label, (BatchLength))
            #!!!resize the data, this should not be here...just for testing
            for i in range(BatchLength):
                if Size[2] == 1:
                    InData[i, :, :, :] = np.reshape(cv2.resize(Data[i, :, :, :], (Size[0], Size[1])), (Size[0], Size[1], Size[2]))
                else:
                    InData[i, :, :, :] = cv2.resize(Data[i, :, :, :], (Size[0], Size[1]))


            # execute teh session
            Summary, _, Acc, L, P = Sess.run([SummaryOp, Optimizer, Accuracy, Loss, OutMaps],
                feed_dict = {InputData: InData, InputLabels: Label, KeepProb: Dropout})

            # print loss and accuracy at every 10th iteration
            if (Step % 10) == 0:
                # train accuracy
                print("Iteration: " + str(Step))
                print("Accuracy:" + str(Acc))
                print("Loss:" + str(L))

            #independent test accuracy
            if (Step%EvalFreq) == 0:
                TotalAcc = 0;
                for i in range(0, TestData.shape[0], BatchLength):
                    if TestData.shape[0] - i < 25:
                        break
                    Data = TestData[i : (i + BatchLength)]
                    InData = np.zeros((BatchLength, Size[0], Size[1], Size[2]))
                    for j in range(BatchLength):
                        if Size[2] == 1:
                        	InData[j, :, :, :] = np.reshape(cv2.resize(Data[j, :, :, :], (Size[0], Size[1])), (Size[0], Size[1], Size[2]))
                        else:
                                InData[j, :, :, :] = cv2.resize(Data[j, :, :, :], (Size[0], Size[1]))
                    Label = TestLabels[i : (i + BatchLength)]
                    Label = np.reshape(Label, (BatchLength))
                    P = Sess.run(Pred, feed_dict = {InputData: InData})
                    for j in range(len(P)):
                        if P[j] == Label[j]:
                            TotalAcc += 1

                print("Independent Test set: "+ str(float(1. * TotalAcc) / TestData.shape[0]))
            #print("Loss:" + str(L))

            SummaryWriter.add_summary(Summary,Step)
            Step += 1

        #print('Saving model...')
        #print(Saver.save(Sess, "./saved/alexnet/model"))

    print("Optimization Finished!")
    print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)
