'''
W1 = np.random.randn(num_inp,num_hid_1) # W1 is hidden units' weights for layer 1. each column is one hidden unit
B1 = np.random.randn(1,num_hid_1) # B1 is the biases for layer 1. one bias for each hidden unit. 
W2 = np.random.randn(num_hid_1,num_hid_2)
B2 = np.random.randn(1,num_hid_2)
W3 = np.random.randn(num_hid_2,1) # weights for output neuron
B3 = np.random.randn(1,1) # bias for output neuron

# ONE INSTANCE
z1 = x @ W1 + B1 # z is the weighted inputs. vector that will go thru sigmoid to become layer's output. 
h1 = sigmoid(z1) # h1 is the final output for layer 1

grad_H4_wrt_H3 = jacobians[-1]
grad_H3_wrt_H2 = jacobians[-2]
grad_H2_wrt_H1 = jacobians[-3]
grad_H1_wrt_H0 = jacobians[-4]

grad_H4_wrt_H3 = jacobians[-1]
grad_H4_wrt_H2 = jacobians[-1] @ jacobians[-2]
grad_H4_wrt_H1 = jacobians[-1] @ jacobians[-2] @ jacobians[-3]
grad_H4_wrt_H0 = jacobians[-1] @ jacobians[-2] @ jacobians[-3] @ jacobians[-4]

grad_H4_wrt_H3 == cumul_jacob[-1]
grad_H4_wrt_H2 == cumul_jacob[-2]
grad_H4_wrt_H1 == cumul_jacob[-3]
grad_H4_wrt_H0 == cumul_jacob[-4]

grad_H4_wrt_H3 = sigmoidp(z_s[4]) * W[4]
grad_H3_wrt_H2 = sigmoidp(z_s[3]) * W[3]
grad_H2_wrt_H1 = sigmoidp(z_s[2]) * W[2]
grad_H4_wrt_H2 = grad_H4_wrt_H3 @ grad_H3_wrt_H2

grad_H4_wrt_H2 == jacobians[4] @ jacobians[3]

layer 2 has 3 neurons
layer 1 has 4 neurons
|  h21_wrt_h11   h21_wrt_h12   h21_wrt_13   h21_wrt_14  |
|  h22_wrt_h11   h22_wrt_h12   h22_wrt_13   h22_wrt_14  |
|  h23_wrt_h11   h23_wrt_h12   h23_wrt_13   h23_wrt_14  |


'''


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
X = np.array([[0,0,1,1],[0,1,0,1]]).T # X is a matrix of inputs for all instances
Y = np.array([[0,1,1,0]]).T # Y is a matrix of all outputs for all instances
num_inp = 2
eta = 0.2

# init weights 
np.random.seed(0)
#             layer:      0 1 2 3 4
#hidden_structure = [num_inp,4,3,2,1] # 2 input units in layer 0, 4 units in layer 1, 3 in 2, 2 in 3, 1 output unit in last layer
hidden_structure = [num_inp,4,4,1]
# need a list of matrices that are: 4x num_inp, 3x4, 2x3, 1x2 since itll be W @ x
W = [None] # so that weights of layer i will be at index i
B = [None]
for i in range(1, len(hidden_structure)):
    W.append(np.random.uniform(size=[hidden_structure[i], hidden_structure[i-1]]))
    B.append(np.random.uniform(size=[hidden_structure[i], 1]))


losses = []
for i in range(32_000):

    # forward pass 
    x = X[i%4].reshape([2,1]) # x is a column vector of inputs for one instance
    y = Y[i%4].reshape([1,1]) # y is the desired output for one instance
    # intermediate stuff 
    input = x
    a_s = [None] # add None first so that layer i will be at index i
    z_s = [None]
    h_s = [x]
    jacobians = [None] # we don't need jacobians[1] because that's wrt to input, x
    for i in range(1, len(hidden_structure)):
        a = W[i] @ input    # matmul
        z = a + B[i]        # add bias
        h = sigmoid(z)      # hidden output
        input = h           # propagate this to next layer's input 
        grad_Hi_wrt_Himinus1 = sigmoidp(z) * W[i] 
        jacobians.append(grad_Hi_wrt_Himinus1)
        a_s.append(a)
        z_s.append(z)
        h_s.append(h)


    # calculate cost
    error = y - input
    losses.append(np.sum(error ** 2))


    # calculate gradient, sum over paths in the graph
    cumul_jacob = [None for i in range(len(jacobians))]
    cumul_jacob[-1] = np.ones([1,1]) # the last is jacobian grad_H4_wrt_H4
    cumul_jacob[-2] = jacobians[-1] # the next is  grad_H4_wrt_H3
    for i in range(-2, -len(jacobians), -1):
        cumul_jacob[i-1] = cumul_jacob[i] @ jacobians[i] # for example, grad_H4_wrt_H2 = grad_H4_wrt_H3 @ grad_H3_wrt_H2


    # update weights 
    for i in range(1, len(hidden_structure)):
        common = cumul_jacob[i].T * sigmoidp(W[i] @ h_s[i-1] + B[i])
        W[i] += eta * error * common @ h_s[i-1].T # not 100% sure about order of operations here but seems to work
        B[i] += eta * error * common
        
losses[-20:]
plt.plot(losses)
plt.ylim(0, 2)
plt.show()

# NOTE: computation graphs will become relevant once i treat the derivative of H4_wrt_H3, the entire matrix, as one node
# rather than a bunch of interconnected nodes 