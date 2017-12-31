# A toy example of using Recurrent Neural Network (RNN)
# RNN state recurrence relation: can by any differentiable function f
# S_t = tanh(S_t-1 * W + X_t * U), where S_t and X_t denote the state and input at step t.
# This toy example trains a basic RNN to count the number of ones in the iput and output the result at the end of the sequence:
# In:  (0, 0, 0, 0, 1, 0, 1, 0, 1, 0)
# Out:  3 

import numpy as np

# Output weight is set to 1
# Linear model is used because of the nature of the problem
def step(s, x, U, W):
	return x * U + s * W
	
# Training using backpropagation through time
# The forward pass
def forward(X, U, W):
	# Initialize the state activation for each sample along the sequence
	numSamples, sequenceLength = np.shape(X)
	S = np.zeros((numSamples, sequenceLength+1))
	# Update the states over the sequence
	for t in range(0,sequenceLength):
		S[:,t+1]=step(S[:,t}, X[:,t], U, W)		# step function
	return S

# The cost function- sum squared error	
def costFun(targets, y):
	cost = np.sum((targets-y)**2)
	return cost

# The backward pass
def backward(X, S, targets, W):
	sequenceLength = np.shape(X)[1]
	# Compute gradient of output
	y = S[:, -1]	# Output y is last activation of sequence
	# Gradient w.r.t. cost functionat final state
	gS = 2.0 * (y - targets)
	# Accumulate gradients backwards
	gU, gW = 0, 0 	# Initialization
	for k in range(sequenceLength, 0, -1):
		# Compute the parameter gradients and accumulate the results
		gU += np.sum(gS * X[:,k-1])
		gW += np.sum(gS * S[:,k-1])
		# Compute the gradient at the output of the previous layer
		gS = gS * W
	return gU, gW
	
# Gradient descent to train the network
def training(X, targets)
	learningRate = 0.0005
	numIterations = 10000
	# Set initial parameters
	parameters = (-2, 0)	# (U, W)
	# Perform iterative gradient descent
	for k in range(numIterations):
		# Perform forward and backward pass to get the gradients
		S = forward(X, parameters(0), parameters(1))
		gradients = backward(X, S, targets, parameters(1))
		# Update each parameter by u = u - (gradient * learningRate)
		parameters = ((p - gp * learningRate)
					for p, gp in zip(parameters,gradients))
	return parameters 
