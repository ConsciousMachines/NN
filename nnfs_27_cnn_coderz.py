import numpy as onp
from PIL import Image as im
import os


# load data
def load_data():
        
    train_dir = r'../input/940-data-1/train/train'
    test_dir = r'../input/940-data-1/test/test'
    labels_dir = r'../input/940-data-1/train_labels.csv'

    train_dir = r'C:\Users\pwnag\Desktop\sup\deep_larn\train\train'
    test_dir = r'C:\Users\pwnag\Desktop\sup\deep_larn\test\test'
    labels_dir = r'C:\Users\pwnag\Desktop\sup\deep_larn\train_labels.csv'

    labels = onp.loadtxt(labels_dir, skiprows=1,dtype=onp.int32,delimiter=',')[:,1]
    Y = onp.zeros([50_000,10])
    Y[onp.arange(50_000), labels] = 1 # one hot encode

    def load_pics(dir):
        files = os.listdir(dir)
        X = onp.zeros([len(files),32,32,3], dtype = onp.uint8)
        for i in range(len(files)):
            real_index = int(files[i].split('/')[-1].split('.')[0])
            X[real_index,:,:,:] = onp.array(im.open(os.path.join(dir, files[i])), dtype = onp.uint8)
        return X
    X = load_pics(train_dir)
    Xt = load_pics(test_dir)
    return X, Y, labels, Xt


_X, _Y, labels, X_test = load_data()





# https://coderzcolumn.com/tutorials/artifical-intelligence/jax-guide-to-create-convolutional-neural-networks#2
# spent a few hours on a bug that batch size 20 gave errors but not batch size of 2^N -.-
import jax
from jax.example_libraries import stax, optimizers
from jax import numpy as np
import time

X = np.array(_X, dtype=np.float32) / 255.0
Y = np.array(_Y, dtype=np.float32)

# to try: batch norm, pools, 
conv_init, conv_apply = stax.serial(
    stax.Conv(32,(3,3), padding="SAME"),
    stax.Relu,
    stax.Conv(16, (3,3), padding="SAME"),
    stax.Relu,

    stax.Flatten,
    stax.Dense(10),
    stax.LogSoftmax
)


@jax.jit
def CrossEntropyLoss(wts, x, y):
    reg = 0
    for w in wts:
        if w: 
            reg += np.sum(w[0] * w[0]) + np.sum(w[1] * w[1])
    return - np.sum(y * conv_apply(wts, x)) + 0.01 * reg


def accuracy(wts):
    preds = []
    num_batches = (50_000 // batch_size) + 1
    for i in range(num_batches):
        _start = i * batch_size
        _end = np.minimum((i + 1) * batch_size, 50_000)
        preds.append(conv_apply(wts, X[_start:_end,:,:,:]))
    return np.sum(np.argmax(np.concatenate(preds).squeeze(), axis=1) == labels)


@jax.jit
def update(opt_state, __idx):
    x, y = X[__idx], Y[__idx] 
    g = jax.grad(CrossEntropyLoss)(opt_get_weights(opt_state), x, y)
    opt_state = opt_update(_e, g, opt_state)
    return opt_state, None


batch_size = 256
epochs = 25
learning_rate = np.array(1/1e4)
mk = jax.random.PRNGKey(42) # master key 
ks = jax.random.split(mk, epochs) # one key per epoch to generate random indices  
wts = conv_init(mk, (batch_size,32,32,3))[1]
for w in wts:
    if w:
        w, b = w
        print(f"Weights : {w.shape}, Biases : {b.shape}")


opt_init, opt_update, opt_get_weights = optimizers.sgd(learning_rate)
opt_state = opt_init(wts)


for _e in range(epochs):


    # generate a permutation of indices for any batch size 
    num_batches = (50_000 // batch_size) + 1
    perm = np.expand_dims(jax.random.permutation(ks[_e], 50_000), axis = 0)
    more = jax.random.choice(ks[_e], 50_000, [1, num_batches * batch_size - 50_000])
    indices = np.concatenate([perm, more],1).reshape([num_batches, batch_size])


    s = time.time()
    
    #for i in range(num_batches):
    #    __idx = indices[i]
    #    x, y = X[__idx], Y[__idx] 
    #    g = jax.grad(CrossEntropyLoss)(opt_get_weights(opt_state), x, y)
    #    opt_state = opt_update(_e, g, opt_state)

    opt_state, _ = jax.lax.scan(update, opt_state, indices) # fold update function over indices
    

    wts = opt_get_weights(opt_state)
    print(f"T: {time.time() - s}\tAcc : {accuracy(wts)}")















