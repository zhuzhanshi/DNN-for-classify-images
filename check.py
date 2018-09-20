#!/usr/bin/env python
# coding:utf8
from dnn import Deep_neural_network
import numpy as np
from dnn_app_utils_v3 import *
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

def data(num):
	x = []
	y = []
	for _ in range(num):
		
		seed = np.random.randint(0,2)
		if seed == 0:
			x.append(np.random.standard_normal(10))
			y.append(1)
		else:
			x.append(np.random.standard_cauchy(10))
			y.append(0)
	return x, y

def data_set(num ):
	X = []
	Y = []
	for i in range(num):
		x, y = data(16)
		x = np.array(x).T
		y = np.array(y).T
		X.append(x)
		Y.append(y)
	return X, Y



if __name__ == '__main__':
	#X, Y = data_set(100)

	train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
	index = 10
	plt.imshow(train_x_orig[index])
	m_train = train_x_orig.shape[0]
	num_px = train_x_orig.shape[1]
	m_test = test_x_orig.shape[0]

	print ("Number of training examples: " + str(m_train))
	print ("Number of testing examples: " + str(m_test))
	print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
	print ("train_x_orig shape: " + str(train_x_orig.shape))
	print ("train_y shape: " + str(train_y.shape))
	print ("test_x_orig shape: " + str(test_x_orig.shape))
	print ("test_y shape: " + str(test_y.shape))

	# Reshape the training and test examples 
	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

	# Standardize data to have feature values between 0 and 1.
	train_x = train_x_flatten/255.
	test_x = test_x_flatten/255.

	print ("train_x's shape: " + str(train_x.shape))
	print ("test_x's shape: " + str(test_x.shape))

	X = train_x
	Y = train_y
	print(Y)


	neural_network = Deep_neural_network(dim=[12288,7,1],batch_size=209)
	neural_network.initialize_parameter()


	for epoch in range(40000):
		neural_network.input = X
		neural_network.label = Y
		#print(neural_network.input)
		neural_network.forward_propagation()
		neural_network.compute_cost()
		neural_network.backward_propagation()
		neural_network.update_parameters()
		if epoch%100 == 0:
			neural_network.input = test_x
			neural_network.label = test_y
			neural_network.forward_propagation()
			neural_network.prediction_accuracy()
			print(neural_network.cost,neural_network.accuracy)
		