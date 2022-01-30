# I took apart Nielsen's NN code. it turned out to be very similar to what I made
# I thought there was some crazy matrix algebra for batching. no. its just a cheap for loop
import pickle
import gzip
import random
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidp(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoidq(x):
    return x * (1 - x)

def one_hot_encode(j):
    e = np.zeros((10))
    e[j] = 1.0
    return e

# load data
f = gzip.open(r'C:\Users\pwnag\Desktop\sup\nielsen\mnist.pkl.gz', 'rb')
_tr, _va, _te = pickle.load(f, encoding = "latin1")
f.close()
tr_x = [np.reshape(x, (784, 1)) for x in _tr[0]] # reshape x's
va_x = [np.reshape(x, (784, 1)) for x in _va[0]]
te_x = [np.reshape(x, (784, 1)) for x in _te[0]]
tr_y = np.array([one_hot_encode(y) for y in _tr[1]]) # one-hot-encode the y's
tr_data = list(zip(tr_x, tr_y))
va_data = list(zip(va_x, _va[1])) # list of tuples of (x,y)
te_data = list(zip(te_x, _te[1]))

def feedforward(a):
    for b, w in zip(B, W):
        a = sigmoid(w @ a + b)
    return a

###############################################################################
###############################################################################




# params
batch_size = 9
eta = 3.0
np.random.seed(1)
num_inp = 784
num_out = 10
# layer                   0 1 2 3
hidden_structure = [num_inp,30,num_out]
B = [None] + [np.random.randn(y, 1) for y in hidden_structure[1:]]
W = [None] + [np.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]


__id3 = np.zeros([batch_size, num_out, num_out])
__idx = np.arange(num_out)
__id3[:,__idx,__idx] = 1
losses = []


#for _e in range(30): # for each epoch
_e = 0

random.seed(10)
random.shuffle(tr_data) # shuffle data 
mini_batches = [tr_data[k:k+batch_size] for k in range(0, len(tr_data), batch_size)] # list of mini_batches


#for mini_batch in mini_batches: # loop over mini batches
mini_batch = mini_batches[0]

mini_batch[0]
type(mini_batch[0][1])
# list of tuples of numpy arrays 
# [(x,y), (x,y), (x,y), (x,y), (x,y)]

# pick 10 random indices out of 50000
__idx = np.random.choice(np.arange(50000),10)
_tr[0][__idx,:].shape
tr_y[__idx,:].shape

__b = 0
batch_x = _tr[0][__b*batch_size:(__b + 1)*batch_size,:].T
batch_y = tr_y[__b*batch_size:(__b + 1)*batch_size,:].T

# forward pass 
input = batch_x
z_s = [None]
h_s = [batch_x]
for j in range(1, len(hidden_structure)):
    z = W[j] @ input + B[j]        # weighted input
    input = sigmoid(z)                 # hidden output
    z_s.append(z)
    h_s.append(input)


# calculate cost
error = input - batch_y
losses.append(np.sum(error ** 2))


jacobians = [np.einsum('ij,il->lij',W[j].T, sigmoidq(h_s[j-1])) for j in range(2, len(hidden_structure))]
cumul_jacob = [None for i in jacobians] + [__id3] 
for j in range(-1,-len(jacobians)-1,-1):
    cumul_jacob[j-1] = np.einsum('pij,pjk->pik', jacobians[j], cumul_jacob[j]) # matmul in inner 2 dimensions


# update weights 
common = (eta / batch_size) * error * sigmoidq(h_s[-1]) # batch_size scalars
for j in range(1, len(hidden_structure)):
    B[j] -= np.expand_dims(np.einsum('lj,jkl->k', common, cumul_jacob[j-1]), 1)
    W[j] -= np.einsum('ij,jhi,wj->hw', common, cumul_jacob[j-1], h_s[j-1])
