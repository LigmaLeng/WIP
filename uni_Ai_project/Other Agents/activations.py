from layers import Layer
import cupy as np


class Activation(Layer):
    def __init__(self, g, g_prime, name="default"):
        super().__init__()
        """
        The activation layer that takes in the weighted vector of inputs denoted by z and applies the activation function g
        where g(Z) = g(Theta . Phi + b)
        """
        self.name = name
        self.g = g
        self.g_prime = g_prime

    def forward(self, z):
        self.z = z
        return self.g(self.z)
    
    def backpropagate(self, dE_dV, optimiser=None):
        """
        [dE_dz]
        is the derivitive of the error with respect to its input,
        which is conversely [dE_dV_l-1] or dE_dV of the previous layer to be used to update its parameters
        dE_dV is denoted here as the derivitive of the error with respect to the output of the activation layer
        following the same chain rules above:
            dE_dZ = dE_dV * dV_dZ
        
        and since the output of the activation layer is V = g(Z), the derivative of V with respect to Z is g_prime
        hence:
            dE_dZ = dE_dV (.) g_prime(Z)
        where (.) computes the Hadamard product/element-wise product of 2 matrices
        """
        return np.multiply(dE_dV, self.g_prime(self.z))
    
class Tanh(Activation):
    def __init__(self):
        
        tanh = lambda x: np.tanh(x)
        
        tanh_prime = lambda x: 1 - (np.tanh(x) ** 2)
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        
        sigma = lambda x: 1.0 / (1.0 + np.exp(x))
        
        sigma_prime = lambda x: sigma(x) - (1 - sigma(x))
        super().__init__(sigma, sigma_prime)


class ReLU(Activation):
    def __init__(self):
        
        relu = lambda x: np.maximum(x, 0)
        
        relu_prime = lambda x:  np.greater(x, 0) * 1.0
        super().__init__(relu, relu_prime)

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        expo = np.exp(z)
        # self.activation = expo / np.sum(expo)
        return expo / np.sum(expo)

    def backpropagate(self, dE_dZ, optimiser=None):
        # n = np.shape(self.activation)[0]
        # return np.dot( (np.identity(n) - self.activation.T) * self.activation, dE_dV)
        return dE_dZ