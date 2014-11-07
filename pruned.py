import numpy as np
import numpy.random as rn
import math as mt

class mlp(object):
	"""initalize multilayer perceptron
		inputs: a matrix  n * d where:
			n = number of input nodes
			d = number of data points

		targets: a matrix  t * 1 where:
			t = number of data points
	"""
	def __init__(self, inputs, targets, hidden_nodes_count, eta):
		self.inputs = inputs
		self.eta = eta
		self.targets = targets
		self.hidden_nodes_count = hidden_nodes_count
		self.init_weights()
		self.add_bias()

	def init_weights(self):
		# get number of input nodes
		self.input_nodes_count = self.inputs.shape[0]
		# w1 is the weights from the input layer to the hidden layer
		# a matrix of h * n where:
		# n = number of input nodes
		# h = number of hidden nodes
		self.w1 = rn.rand(self.hidden_nodes_count, self.input_nodes_count+1)/10
		# add weights for lateral connections
		self.lateral = rn.rand(self.hidden_nodes_count, self.hidden_nodes_count)/10
		# get number of input nodes
		self.output_nodes_count = self.targets.shape[0]
		# w2 is the weights from the hidden layer to the output layer
		# a matrix of o * h where:
		# o = number of output nodes
		# h = number of hidden nodes
		self.w2 = rn.rand(self.output_nodes_count, self.hidden_nodes_count+1)/10

	def add_bias(self):
		# add bias input for all data points
		self.points_count = self.inputs.shape[1]
		bias_input = np.ones((1,self.points_count))
		self.inputs = np.vstack((bias_input, self.inputs))

	def display(self):
		print("inputs", self.inputs)
		print("w1", self.w1)
		print("w2", self.w2)
		print("targets", self.targets)
		print("y1", self.y1)
		print("y2", self.y2)

	def online_train(self):
		for j in range(0,200):
			for i in range(0,self.points_count):
				self.online_forward_pass(i)
				self.back_propagate(i)

	# NOTE : pruning requires pibolar
	def segmoid(net):
		out = 0.5 * ( (1 - np.exp(-net))/(1 + np.exp(-net)) )
		return out
	


	def online_forward_pass(self, index):
		self.y1_net = np.dot(self.w1,self.inputs[:,[index]])
		self.y1 = mlp.segmoid(self.y1_net)
		for i in range(1,self.hidden_nodes_count):
				lateral_i = np.dot(self.lateral[i,:i],self.y1[:i])
				print(lateral_i.shape)
				self.y1_net[i] += lateral_i
				self.y1[i] = mlp.segmoid(self.y1_net[i])

		self.y1 = np.vstack((np.array([[1]]),self.y1))
		self.y2_net = np.dot(self.w2,self.y1)
		self.y2 = mlp.segmoid(self.y2_net)

	def back_propagate(self, index):
		error = self.targets[:,[index]] - self.y2
		delta1 = error * self.y2 * (1 - self.y2)
		dw = self.eta * np.dot(delta1, self.y1.transpose())
		self.w2 += dw
		delta2 = np.dot(self.w2.transpose(), delta1) * (self.y1) * (1 - self.y1)
		delta2 = delta2[1:,]
		dw = self.eta * np.dot(delta2,self.inputs[:,[index]].transpose())
		self.w1 += dw


inputs = np.array([[-1,1],[-1,-1],[0,0],[1,0]]).transpose()
targets = np.array([[1,1,0,0],[0,0,1,1]])

p = mlp(inputs, targets, 4, 1)
p.online_train()