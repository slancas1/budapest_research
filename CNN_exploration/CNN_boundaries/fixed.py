#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import cv2
import copy
from random import randint

# parameters
InputImage = 'avergra2.bmp'
InputImg = cv2.cvtColor(cv2.imread(InputImage), cv2.COLOR_BGR2GRAY) # reads in the image

# this is the threshold template
A = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
B = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
z = 0

def imageToCNN(img): # converts the numpy array representing the image to the CNN representation
	# we have a scale from 0 to 255 and want to convert to a scale from 1 to -1
	# flip (divide by -1)
	# divide up the interval (length of 255 to 2)
	# shift the interval to align
	return -1 * (img / 128.0 - 1.0)

def CNNToImage(img): # converts the CNN representation to the numpy array
	# does the opposite of the steps listed in the imageToCNN function
	return ((img * -1.0) + 1.0) * 128.0

# function that creates the y graph (sets values to -1 and 1 if they are out of those bounds)
def nonLinearity(img):
	# brings in a matrix / image
	# will threshold the image below -1 and above 1
	Out = copy.deepcopy(img)
	Out[np.where(img < -1)] = -1
	Out[np.where(img > 1)] = 1
	return Out

def CellEquation(State, t, Input, A, B, z):
	# x' = -x + A*y + B*u + z
	# this is for a grid
	StateMat = np.reshape(State, [Input.shape[0], Input.shape[1]])

	ExpandedM = np.zeros((Input.shape[0] + 2, Input.shape[1] + 2))
	ExpandedM[1 : ExpandedM.shape[0] - 1, 1 : ExpandedM.shape[1] - 1] = Input[:][:]
	ExpandedM[0][:] = randint(-1, 1)
	ExpandedM[:][0] = randint(-1, 1)
	ExpandedM[ExpandedM.shape[0] - 1][:] = randint(-1, 1)
	ExpandedM[:][ExpandedM.shape[1] - 1] = randint(-1, 1)

	ExpandedState = np.zeros((StateMat.shape[0] + 2, StateMat.shape[1] + 2))
	ExpandedState[1 : ExpandedState.shape[0] - 1, 1 : ExpandedState.shape[1] - 1] = StateMat[:][:]
	ExpandedState[0][:] = randint(-1, 1)
	ExpandedState[:][0] = randint(-1, 1)
	ExpandedState[ExpandedState.shape[0] - 1][:] = randint(-1, 1)
	ExpandedState[:][ExpandedState.shape[1] - 1] = randint(-1, 1)

	Output = np.zeros(InputImg.shape)
	Nonlinear = nonLinearity(ExpandedState)
	for x in range(1, InputImg.shape[0] + 1):
		for y in range(1, InputImg.shape[1] + 1):
			negX = np.multiply(-1, ExpandedState[x, y])
			StateRegion = Nonlinear[(x - 1) : (x + 2), (y - 1) : (y + 2)]
			Ay = np.sum(np.multiply(A, nonLinearity(StateRegion)))
			Bu = np.sum(np.multiply(B, ExpandedM[(x - 1) : (x + 2), (y - 1) : (y + 2)]))
			Output[x - 1, y - 1] = negX + Ay + Bu + z
	Output = np.reshape(Output, [Output.shape[0] * Output.shape[1]])
	return Output

# uses Euler's method to produce a faster solution
def Euler(func, State, h, NumSteps, Input, A, B, z):
	States = np.zeros((NumSteps, State.shape[0]))
	States[0, :] = State
	for i in range (1, NumSteps):
		States[i, :] = States[i - 1, :] + np.multiply(h, func(States[i - 1, :], i * h, Input, A, B, z))
	return States

TimeSteps = np.arange(0, 2.0, 0.01)

Input = imageToCNN(InputImg)
# reshape the input into one big vector
InitialState = np.zeros(Input.shape)
InitialState = copy.deepcopy(Input)
InitialState = np.reshape(InitialState, [InputImg.shape[0] * InputImg.shape[1]])

# r = scipy.integrate.odeint(CellEquation, InitialState, TimeSteps)
r = Euler(CellEquation, InitialState, 0.1, 30, Input, A, B, z)

OutImg = np.reshape(r[-1, :], [Input.shape[0], Input.shape[1]])
OutImg = nonLinearity(OutImg)
OutImg = CNNToImage(OutImg)

cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Input Image', 300, 300)
cv2.imshow('Input Image', InputImg)

cv2.namedWindow('Output Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output Image', 300, 300)
cv2.imshow('Output Image', OutImg)

cv2.waitKey(0)
