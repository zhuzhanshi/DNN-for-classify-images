#!/usr/bin/env python
# coding:utf8

import numpy as np
np.random.seed(1)

class Deep_neural_network:
	
	def __init__(self, input=None, label=None, dim=None, batch_size=16, learning_rate=0.0075):
		self.input = input
		self.label = label
		self.dim = dim
		self.batch_size = batch_size
		self.prediction = None
		self.cost = None
		self.learning_rate = learning_rate
		self.accuracy = None
		self.W = []
		self.b = []
		self.A = []
		self.Z = []
		self.dW = []
		self.db = []
		self.dA = []
		self.dZ = []

	def relu(self,z):
		return np.maximum(z,0)

	def sigmoid(self,z):
		denominator = 1 + np.exp(-z)
		return 1/denominator

	def relu_backward(self,z):
		z[z <= 0] = 0
		z[z > 0] = 1
		return z
	
	def sigmoid_backward(self,z):
		pass

	def initialize_parameter(self):

		num = len(self.dim)
		#print(num)
		for i in range(num-1):
			#print(i)
			layer_W = [self.dim[i+1],self.dim[i]]
			layer_b = [self.dim[i+1],1]
			self.W.append(np.random.standard_normal(layer_W)*0.01)
			self.b.append(np.zeros(layer_b)+0.01)
			self.A.append(np.zeros([self.dim[i+1], self.batch_size]))
			self.Z.append(np.zeros([self.dim[i+1], self.batch_size]))
			self.dW.append(np.zeros(layer_W))
			self.db.append(np.zeros(layer_b))
			self.dA.append(np.zeros([self.dim[i+1], self.batch_size]))
			self.dZ.append(np.zeros([self.dim[i+1], self.batch_size]))

	def cell_forward_propagation(self, deep_num = None, activation = 'relu'):
		if activation == 'relu':
			##print(deep_num)
			#print(self.Z[deep_num].shape, self.W[deep_num].shape,self.b[deep_num].shape,self.A[deep_num+1].shape)
			self.Z[deep_num] = np.dot(self.W[deep_num],self.A[deep_num]) + self.b[deep_num]
			self.A[deep_num+1] = self.relu(self.Z[deep_num])
		elif activation == 'sigmoid':
			#print(deep_num)
			self.Z[deep_num] = np.dot(self.W[deep_num],self.A[deep_num]) + self.b[deep_num]
			self.prediction = self.sigmoid(self.Z[deep_num])

	def forward_propagation(self):
		self.A[0] = self.input
		num = len(self.dim) - 1
		for num_deep in range(num-1):
			self.cell_forward_propagation(num_deep,activation='relu')
		self.cell_forward_propagation(num-1,activation='sigmoid')

	def compute_cost(self):
		m = self.batch_size
		#print(self.label.shape)
		#print(self.prediction.shape)

		cost = 0 - np.sum(np.multiply(np.log(self.prediction),self.label) + np.multiply(np.log(1-self.prediction),1 - self.label))
		#cost = 0 - np.sum(self.label*np.log(self.prediction) + (1-self.label)*np.log(1-self.prediction))
		self.cost = cost/m


	def cell_backward_propagation(self, num_deep = None, activation = 'relu'):
		m = self.batch_size
		if activation == 'relu':
			#print(num_deep)
			self.dZ[num_deep] = np.multiply(self.dA[num_deep+1], self.relu_backward(self.Z[num_deep]))
			self.dW[num_deep] = np.dot(self.dZ[num_deep], self.A[num_deep].T)/m
			self.db[num_deep] = np.sum(self.dZ[num_deep], axis=1, keepdims=True)/m
			self.dA[num_deep] = np.dot(self.W[num_deep].T, self.dZ[num_deep])
		elif activation == 'sigmoid':
			#print(num_deep)
			self.dZ[num_deep] = self.prediction - self.label
			self.dW[num_deep] = np.dot(self.dZ[num_deep], self.A[num_deep].T)/m
			self.db[num_deep] = np.sum(self.dZ[num_deep], axis=1, keepdims=True)/m
			self.dA[num_deep] = np.dot(self.W[num_deep].T, self.dZ[num_deep])


	def backward_propagation(self):
		num = len(self.dim) - 1
		self.cell_backward_propagation(num-1,activation='sigmoid')
		for num_deep in reversed(range(num-1)):
			self.cell_backward_propagation(num_deep)

	def update_parameters(self):
		num = len(self.dim) - 1

		for num_deep in range(num):
			#print(self.dW[num_deep].shape)
			self.W[num_deep] = self.W[num_deep] - self.learning_rate*self.dW[num_deep]
			self.b[num_deep] = self.b[num_deep] - self.learning_rate*self.db[num_deep]
	
	def prediction_accuracy(self):
		m = self.batch_size
		pre = np.zeros(self.label.shape)
		#print(self.prediction)
		pre[self.prediction<0.5] = 0.0
		pre[self.prediction>=0.5] = 1.0
		accuracy = float((np.dot(self.label,pre.T) + np.dot(1-self.label,1-pre.T))/float(self.label.size)*100)
		self.accuracy = accuracy

	def neural_network_model(self):
		X = self.input

		self.forward_propagation()






if __name__ == '__main__':
	pass