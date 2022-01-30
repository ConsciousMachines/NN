import pickle
import gzip
import jax
import jax.numpy as np
import numpy as onp

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidp(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoidq(x):
    return x * (1 - x)

def one_hot_encode(j):
    e = onp.zeros((10))
    e[j] = 1.0
    return e

# load data
f = gzip.open(r'C:\Users\pwnag\Desktop\sup\nielsen\mnist.pkl.gz', 'rb')
_tr, _va, _te = pickle.load(f, encoding = "latin1")
f.close()
tr_x = [onp.reshape(x, (784, 1)) for x in _tr[0]] # reshape x's
va_x = [onp.reshape(x, (784, 1)) for x in _va[0]]
te_x = [onp.reshape(x, (784, 1)) for x in _te[0]]
tr_y = onp.array([one_hot_encode(y) for y in _tr[1]]) # one-hot-encode the y's
tr_data = list(zip(tr_x, tr_y))
va_data = list(zip(va_x, _va[1])) # list of tuples of (x,y)
te_data = list(zip(te_x, _te[1]))



def _feedforward(params, _x):
    _n = len(hidden_structure)-1
    for _i in range(_n):
        _x = sigmoid(onp.matmul(params[_i + _n], _x) + params[_i]) # TODO  
    return _x


#soy = _feedforward(params, x)
#onp.array([[1,2],[3,4]])
#onp.matmul(,[1,2])


def feedforward(params, _x):
    _n = len(hidden_structure)-1
    for _i in range(_n):
        _x = sigmoid(np.dot(params[_i + _n], _x) + params[_i]) # TODO  
    return _x

def loss(params, x, y): # Cross-entropy loss
    err = feedforward(params, x) - y
    return np.dot(err,err)  #-y * np.log(out) - (1 - y) * np.log(1 - out)

loss_grad = jax.grad(loss)
  

###############################################################################
###############################################################################


# params
eta = 1.0
onp.random.seed(1)
num_inp = 784
num_out = 10
# layer                   0 1 2 3
hidden_structure = [num_inp,30,num_out]
params = [onp.random.randn(y, 1) for y in hidden_structure[1:]] + [onp.random.randn(y, x) for x, y in zip(hidden_structure[:-1], hidden_structure[1:])]


for _e in range(30): # for each epoch

    for _q in range(1000):

        # create batch
        __idx = onp.random.choice(onp.arange(50000))
        x = _tr[0][__idx,:].T.squeeze()
        y = tr_y[__idx,:].T.squeeze()


        #x = X[onp.random.choice(X.shape[0])]          # Grab a single random input
        #y = onp.bitwise_xor(*x)                       # Compute the target output
        grads = loss_grad(params, x, y)
        params = [param - eta * grad for param, grad in zip(params, grads)]

    # test on te_data
    __test_results = [(onp.argmax(_feedforward(params, x)), y) for (x, y) in te_data]
    __evaluate = sum(int(x == y) for (x, y) in __test_results)
    print("Epoch {} : {} / {}".format(_e, __evaluate, len(te_data)))









