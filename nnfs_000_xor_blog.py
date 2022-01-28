import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidp(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoidq(x):
    return x * (1 - x)

# init data
X = np.array([[0,0,1,1],[0,1,0,1]]).T
Y = np.array([[0,1,1,0]]).T

# initialize weights
eta = 0.2
num_input = 2
num_hidden = 2
num_output = 1
np.random.seed(0)
W1 = np.random.uniform(size=(num_input, num_hidden))
W2 = np.random.uniform(size=(num_hidden, num_output))
B1 = np.random.uniform(size=(1,num_hidden))
B2 = np.random.uniform(size=(1,num_output))


losses = []
for i in range(8000):

    # forward pass 
    H = sigmoid(X @ W1 + B1)
    out = sigmoid(H @ W2 + B2)

    # calculate cost
    error = Y - out
    losses.append(np.sum(error ** 2))

    # calculate gradient
    common2 = eta * error * sigmoidp(H @ W2 + B2)
    common1 = common2 @ W2.T * sigmoidp(X @ W1 + B1)

    # update
    W2 += H.T @ common2
    B2 += np.sum(common2, axis = 0)
    W1 += X.T @ common1
    B1 += np.sum(common1, axis = 0)
    if i % 500 == 0:
        print("error: ", losses[i]) 