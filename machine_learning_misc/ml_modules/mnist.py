import cupy as cp
import struct
from eden import Eve
from layers import *
from activations import Sigmoid, Softmax, ReLU
from losses import cross_entropy, cse_softmax_prime

def load():
    with open('train-labels-idx1-ubyte', 'rb')as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = cp.fromfile(labels, dtype=cp.uint8)
    with open('train-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_images = cp.fromfile(imgs, dtype=cp.uint8).reshape(num, 28, 28)
    with open('t10k-labels-idx1-ubyte', 'rb')as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = cp.fromfile(labels, dtype=cp.uint8)
    with open('t10k-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_images = cp.fromfile(imgs, dtype=cp.uint8).reshape(num, 28, 28)
    return train_images, train_labels, test_images, test_labels

def one_hot(y, num_labels=10):
    one_hot_label = cp.zeros((num_labels, y.shape[0]))
    for i, label in enumerate(y):
        one_hot_label[label, i] = 1
    return one_hot_label

x_train, y_train, x_test, y_test = load()
y_train = one_hot(y_train)
x_train = cp.asarray(x_train, dtype=cp.float32)
y_train = cp.asarray(y_train, dtype=cp.float32)
x_test = cp.asarray(x_test, dtype=cp.float32)
y_test = cp.asarray(y_test, dtype=cp.float32)

network = [
        Conv2D((1, 28, 28), (5, 5), 5, name="first"),
        ReLU(),
        Conv2D((5, 24, 24), (3, 3), 5),
        ReLU(),
        Transmogrify((5, 22, 22), (5 * 22 * 22, 1)),
        Dense(5 * 22 * 22, 200),
        Sigmoid(),
        Dense(200, 10),
        Softmax()
    ]

eve = Eve(network, cross_entropy, cse_softmax_prime,
            epochs = 30000, batch_size = 3, alpha = 0.007)
eve.train(x_train=x_train, y_train=y_train)
eve.test(x_test=x_test, y_test=y_test, test_size = 10000)


