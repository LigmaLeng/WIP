from . import xp

listed = ["ReLU", "Softmax", "Sigmoid", "Tanh", "Leaky_ReLU", "Norm_Softmax", "Linear"]

class BaseActivation:
    def __init__(self, g, gradfn):
        """
        The activation layer that takes in the weighted vector of inputs denoted by z and applies the activation function g
        where g(Z) = g(Theta . Phi + b)
        """
        self.g = g
        self.gradfn = gradfn

    def __call__(self, z):
        self.z = z
        return self.g(self.z)

    def grad(self, dY):
        """
        [dL_dz]
        is the derivitive of the error with respect to its input,
        which is conversely [dL_dY_l-1] or dL_dY of the previous layer to be used to update its parameters
        dL_dY is denoted here as the derivitive of the error with respect to the output of the activation layer
        following the same chain rules above:
            dL_dZ = dL_dY * dY_dZ
        
        and since the output of the activation layer is Y = g(Z), the derivative of Y with respect to Z is g'
        hence:
            dL_dZ = dL_dY * g'(Z)
        where * computes the Hadamard product/element-wise product of 2 matrices
        """
        return dY * self.gradfn(self.z)

class Tanh(BaseActivation):
    gain=5/3
    def __init__(self):
        tanh = lambda x: xp.tanh(x)
        tanh_prime = lambda x: 1 - (xp.tanh(x) ** 2)
        super().__init__(tanh, tanh_prime)

class Sigmoid(BaseActivation):
    gain=1
    def __init__(self):
        sigma = lambda x: 1.0 / (1.0 + xp.exp(-x))
        sigma_prime = lambda x: sigma(x) - (1 - sigma(x))
        super().__init__(sigma, sigma_prime)


class ReLU(BaseActivation):
    gain=2**.5
    def __init__(self):
        relu = lambda x: xp.maximum(x, 0)
        relu_prime = lambda x:  xp.greater(x, 0) * 1.0
        super().__init__(relu, relu_prime)

class Leaky_ReLU(BaseActivation):
    gain=(2/(1 + .01**2))**.5
    def __init__(self):
        releak = lambda x: xp.maximum(x*0.01, x)
        releak_prime = lambda x:  xp.where(x > 0, 1, 0.01).astype(xp.float32)
        super().__init__(releak, releak_prime)

class Linear:
    gain=1
    @classmethod
    def __call__(cls, z):
        return z
    
    @classmethod
    def grad(cls, dZ):
        return dZ

class Softmax:
    gain=1
    @classmethod
    def __call__(cls, z):
        expo = xp.exp(z)
        # self.activation = expo / xp.sum(expo)
        return expo / xp.sum(expo, axis=1, keepdims=True)
    
    @classmethod
    def grad(cls, dZ):
        # n = xp.shape(self.activation)[0]
        # return xp.dot( (xp.identity(n) - self.activation.T) * self.activation, dE_dV)
        return dZ
    
class Norm_Softmax:
    gain=1
    @classmethod
    def __call__(cls, z):
        epsilon = 1e-12
        z = z/xp.std(z, axis=1, keepdims=True)
        expo = xp.exp(z)
        return (expo + epsilon)/ (xp.sum(expo, axis=1, keepdims=True) + epsilon)
    
    @classmethod
    def grad(cls, dZ):
        return dZ


