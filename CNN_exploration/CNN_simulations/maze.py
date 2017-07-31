#!/usr/bin/env python2.7

import cv2
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import copy

InputImage = 'Labyrinth.bmp'
InputImg = cv2.cvtColor(cv2.imread(InputImage), cv2.COLOR_BGR2GRAY)

InitImage = 'Init.bmp'
InitImage = cv2.cvtColor(cv2.imread(InitImage), cv2.COLOR_BGR2GRAY)

# convert image data to CNN domain: -1 white, 1 black
def Bitmap2CNN(Img):
	return (-1.0) * (Img / 127.5 - 1.0)

# convert CNN data to image data: 0 black, 255 white
def CNN2Bitmap(Img):
    Out = ((Img * (-1.0)) + 1.0) * 127.5
    return Out.astype(np.uint8)

# implementation of the standard CNN nonlinearity, sturates all values below -1 and above 1
def Nonlinearity(Img):
    Out = copy.deepcopy(Img);
    Out[np.where(Img < -1)] = -1;
    Out[np.where(Img > 1)] = 1;
    return Out

# impelmentation of the CNN state equation
# x'=-x+ ay + bu +z
def CellEquation(State, t, Input, A, B, Z):
    Out = np.zeros(Input.shape)
    Output = Nonlinearity(State)
    for x in range(1, (Input.shape[0] - 1)):
        for y in range(1, (Input.shape[1] - 1)):
            InputRegion = Input[(x - 1) : (x + 2), (y - 1) : (y + 2)]
            OutputRegion = Output[(x - 1) : (x + 2), (y - 1) : (y + 2)]
            Out[x, y] = -1 * State[x, y] + np.sum(np.multiply(OutputRegion, A) + np.multiply(InputRegion, B)) + Z
    return Out

# Euler method to give a fast solution
def Euler(func, State, h, NumSteps, Input, A, B, Z):
    h = 0.1 #timestep
    States = np.zeros((NumSteps, State.shape[0], State.shape[1]))
    States[0, :] = State
    for i in range(1, NumSteps):
        States[i, :, :] = States[i - 1, :, :] + np.multiply(h, func(States[i - 1, :, :], i * h, Input, A, B, Z) )
    return States


Input = Bitmap2CNN(InputImg)
InitState = np.zeros(Input.shape) # initial state as all zeros
InitState = copy.deepcopy(Input) # initial state is the input image

ENDEATER_A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
ENDEATER_B = [[0, 2, 0], [2,9, 2], [0, 2, 0]]
ENDEATER_Z = 7

LOGAND_A = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
LOGAND_B = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
LOGAND_Z = -1

LAB_A = [[0, 2, 0], [2,9, 2], [0, 2, 0]]
LAB_B = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
LAB_Z = 5

TimeSteps = np.arange(0, 2.0, 0.01)

# execute the simulation
InitImage = Bitmap2CNN(InitImage)

# this is the wave propogation way of eliminating the end points, happens in one step -- when A has the numbers and B = 0
r = Euler(CellEquation, Input, 0.1, 80, InitImage, LAB_A, LAB_B, LAB_Z) # this uses my implementaiton, it is faster, manually have to set timestep and iteration number
OutImg = r[-1, :, :]
OutImg = Nonlinearity(OutImg) # we need the output of the last state

# this is the iterative way of eliminating the end points -- when A = 0 and B has the numbers
'''for i in range(0,20):
	r = Euler(CellEquation, InitState, 0.1, 30, Input, ENDEATER_A, ENDEATER_B, ENDEATER_Z) # this uses my implementaiton, it is faster, manually have to set timestep and iteration number
	OutImg = r[-1, :, :]
	OutImg = Nonlinearity(OutImg) # we need the output of the last state

	r = Euler(CellEquation, InitImage, 0.1, 30, OutImg, LOGAND_A, LOGAND_B, LOGAND_Z) # this uses my implementaiton, it is faster, manually have to set timestep and iteration number
	OutImg = r[-1, :, :]
	OutImg = Nonlinearity(OutImg) # we need the output of the last state
	Input = OutImg'''

cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Input Image', 300, 300)
cv2.imshow('Input Image', InputImg)

cv2.namedWindow('Output Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output Image', 300, 300)
cv2.imshow('Output Image', CNN2Bitmap(OutImg))

cv2.waitKey(0)
