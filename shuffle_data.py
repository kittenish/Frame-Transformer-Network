import numpy as np
import random
import json
from read_mat import read_mat

def shuffle(x_train, y_train, num):

	x = np.linspace(0, num - 1, num)
	random.shuffle(x)

	X_train = np.zeros([num, 60, 4096])
	Y_train = np.zeros((num,), dtype=np.int)

	for i in range(num):
		X_train[i,:,:] = x_train[int(x[i]),:,:]
		Y_train[i] = y_train[int(x[i])]

	print 'finish train data shuffle'
	return X_train, Y_train

def shuffle_100(x_train, y_train, num):

	x = np.linspace(0, num - 1, num)
	random.shuffle(x)

	X_train = np.zeros([num, 100, 4096])
	Y_train = np.zeros((num,), dtype=np.int)

	for i in range(num):
		X_train[i,:,:] = x_train[int(x[i]),:,:]
		Y_train[i] = y_train[int(x[i])]

	print 'finish train data shuffle'
	return X_train, Y_train


def shuffle_emotion6(x_train, y_train, position, num):

	print x_train.shape
	print y_train.shape
	print position.shape
	x = np.linspace(0, num - 1, num)
	random.shuffle(x)

	X_train = np.zeros([num, 30, 4096])
	Y_train = np.zeros((num,), dtype=np.int)
	Position = np.zeros([num, 2], dtype=np.int)

	for i in range(num):
		X_train[i,:,:] = x_train[int(x[i]),:,:]
		Y_train[i] = y_train[0,int(x[i])]
		Position[i,:] = position[int(x[i]),:]
	
	theta = np.zeros([num, 2])
	for i in range(num):
		theta[i,0] = (Position[i,1] - Position[i,0]) / 30.0
		theta[i,1] = ((Position[i,1] + Position[i,0]) / 15.0 - 2.0) / 2.0
	
	np.savetxt("/home/g_jiarui/video_spacial/result/1/train/train_position.txt",Position,fmt="%d")
	np.savetxt("/home/g_jiarui/video_spacial/result/1/train/train_theta.txt",theta,fmt="%f")

	print 'finish train data shuffle'
	return X_train, Y_train, Position

def shuffle_conv5(x_train, y_train, num):

	x = np.linspace(0, num - 1, num)
	random.shuffle(x)

	X_train = x_train.copy()
	Y_train = np.zeros((num,), dtype=np.int)

	for i in range(num):
		X_train[i,:,:,:,:] = x_train[int(x[i]),:,:,:,:]
		Y_train[i] = y_train[int(x[i])]

	print 'finish train data shuffle'
	return X_train, Y_train

