from . import xp
from . import BaseLayer
from . import functools as F
from typing import Callable, Any

class Dense(BaseLayer):
    def __init__(self, input_size, output_size, has_bias=True, activation=None, dump_grad=False, w_init="kaiming_normal", name="default"):
        """
        Fully connected layer

        For a minibatch size m
        Dense object is fully connected layer mapping input X to output Y
        where Y = XW + B
        and B is a vector for the biases
        and X is the feature vector

        note: the equation can be modified where Y = WT.X + B
        and the weight matrix would just have to be modified to shape (j x i) to simplify the backward function.
        The various sources follow the WT forward convention
        (not sure why, possibly due to the arrangement of the terms when deriving the chain rule for backpropogation) 

        If output Y is m number of column vectors 1 x j
        and input X is m number of column vectors 1 x i
        The weight vector Theta should then be a matrix of size i x j
        As B measures the bias for every output, it should match the shape of Y : 1 x j
        """
        super().__init__(activation=activation, name=name, has_bias=has_bias, dump_grad=dump_grad)
        self.input_size = input_size
        self.output_size = output_size

        self.W = F.weight_init(w_init, self.input_size, self.output_size, self.activation.gain, size=(self.input_size, self.output_size))
        self.params[f"{self.name}_W"] = self.W
        if self.has_bias:
            self.B = xp.zeros((1, self.output_size), dtype=xp.float32)
            self.params[f"{self.name}_B"] = self.B
        else: self.B = 0
        self.tags.extend(self.params.keys())

    def forward(self, X):
        self.X = X if self.training else None
        return self.activation((X @ self.W) + self.B)
    
    def backpropagate(self, dY, optimiser:Callable):
        """
        ===========================================================================================
        !! DERIVING DIFFERENTIALS !!

        [dL_dX]
        is the derivative of the error with respsect to the input tensor X
        following chain rule if: 

            dL_dX = dL_dY * dY_dX

        then:

            dL_dxi = dL_dY * dY_dxi

        since the gradient of the output of each element yj in Y with respect to each element xi in X
        excludes all other terms that don't include the feature xi, then:

            dY_dX = (d_dX) XW + B
                    = W

        hence by applying the inner product:

            dL_dX = dL_dY . W

        but since the shape of dL_dX should be m x i
        the shape of Theta would have to be transposed from i x j to j x i
        to map the respective input-output mappings, therfore:

            dL_dX = dL_dY . WT

        where the shape of dL_dX is obtained from (m x j) . (j x i)
        ===========================================================================================
        """
        dY = self.activation.grad(dY)
        if not self.dump_grad:
            dX = dY @ self.W.T

        optimiser(self, self.dW(dY), self.dB(dY))

        if not self.dump_grad:
            return dX 
        else:
            del dY
            return 0
        
    def dW(self, dY):
        """
        ===========================================================================================
        [dL_dW]

        is the derivative of the error with respsect to the weight paramaters W.
        following the same chain rules above:

            dL_dW = dL_dY * dY_dW

        then:

            dL_dwij = dL_dyj * dyj_dwij

        since yj is bj + the product of xi and wij,
        the deriative of yj with respect to wij would exclude all terms not including wij,
        therefore:

            dL_dwij = dL_dyj * [(d_dwij) xi * wij + bj]
                    = dL_dyj * xi
            generalised = dL_dY * X

        with matrix multiplication you can get the dot product of the 2 vectors by:
        
            dL_dW = XT . dL_dY

        where X is transposed to yield dL_dW with shape ixj
        ===========================================================================================
        """
        return (self.X.T @ dY)
    
    def dB(self, dY):
        """
        ===========================================================================================
        [dL_dB]

        is the derivative of the error with respsect to the biases vector B.
        following the same chain rules above:

            dL_dB = dL_dY * dY_dB

        since the exponent of bj is 1 and is an addition operation, the deriative of yj with respect to bj is also 1
        hence:

            dL_dB = dL_dY

        ===========================================================================================
        """
        if not self.has_bias: return None
        return xp.sum(dY, axis=0, keepdims=True)

        

