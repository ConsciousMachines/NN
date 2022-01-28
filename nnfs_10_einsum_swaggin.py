# EINSTEIN NOTATION FOR CHADS. does this solve the matrix problem? not really, but I feel empowered. 
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
num_inp = 2
eta = -1.
batch_size = 4


# init weights 
np.random.seed(0)
# layer                   0 1 2 3
hidden_structure = [num_inp,5,3,1]
W = [None] # so that weights of layer i will be at index i
B = [None]
for i in range(1, len(hidden_structure)):
    W.append(np.random.uniform(size=[hidden_structure[i], hidden_structure[i-1]]))
    B.append(np.random.uniform(size=[hidden_structure[i], 1]))


losses = []
for ii in range(2500):


    # forward pass 
    input = X.T
    z_s = [None]
    h_s = [X.T]
    for j in range(1, len(hidden_structure)):
        z = W[j] @ input + B[j]        # weighted input
        h = sigmoid(z)                 # hidden output
        input = h                      # propagate this to next layer's input 
        z_s.append(z)
        h_s.append(h)


    # calculate cost
    error = input - Y.T
    losses.append(np.sum(error ** 2))


    # takes each column of z_s[j-1], multiplies it broadcasting element wise by W[j].T, and stacks results. work it out on paper
    jacobians = [np.einsum('ij,il->lij',W[j].T, sigmoidq(h_s[j-1])) for j in range(2, len(hidden_structure))]
    cumul_jacob = [None for i in jacobians] + [np.ones([batch_size,1,1])] # the last is jacobian grad_H4_wrt_H4
    for j in range(-1,-len(jacobians)-1,-1):
        cumul_jacob[j-1] = np.einsum('pij,pjk->pik', jacobians[j], cumul_jacob[j]) # matmul in inner 2 dimensions


    # update weights 
    common = eta * error * sigmoidq(h_s[-1]) # batch_size scalars
    for j in range(1, len(hidden_structure)):
        B[j] += np.einsum('ij,jkl->kl', common, cumul_jacob[j-1])
        W[j] += np.einsum('qb,bij,ob->io', common, cumul_jacob[j-1], h_s[j-1])




plt.plot(losses)
plt.ylim(0, 2)
plt.show()

