from layers import *
from eden import *
import numpy as np
from losses import mse, mse_prime
from activations import Tanh
import matplotlib.pyplot as plt

def pred(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss_function, loss_prime, x_y_train, epochs=2000, alpha=cp.asarray([0.1])):
    for e in range(epochs):
        err = 0
        cp.random.shuffle(x_y_train)
        for i in range(x_y_train.shape[0]):
            triplet = x_y_train[i]
            x_train = cp.asarray([triplet[0], triplet[1]])
            y_train = cp.asarray(triplet[2])
            x_train = cp.reshape(x_train, (2, 1))
            y_train = cp.reshape(y_train, (1, 1))
            #forward
            output = pred(network, x_train)
            #error
            err += loss_function(y_train, output)
            #backprop
            dE_dV = loss_prime(y_train, output)
            for layer in reversed(network):
                dE_dV = layer.backpropagate(dE_dV, alpha)

        err /= len(x_train)
        print("{}/{}, error={}".format(e + 1, epochs, err))


input = np.reshape([[0,0,0], [0,1,1], [1,0,1], [1,1,0]], (4, 3, 1))
input = cp.asarray(input, dtype=cp.float32)

mlp = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

#training
train(mlp, mse, mse_prime, input)


# testing with decision boundary plot
points = []
y_hat = []
for x in np.linspace(0, 1, 30):
    for y in np.linspace(0, 1, 30):
        points.append((x, y))
for x in range(len(points)):
    x_test = cp.asarray(points[x])
    x_test = cp.reshape(x_test, (2, 1))
    v_hat = cp.asnumpy(pred(mlp, x_test))
    y_hat.append(v_hat[0][0])

points = np.array(points)
plt.style.use('dark_background')
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], y_hat[:], c=y_hat[:], cmap="plasma_r")
plt.show()
    

