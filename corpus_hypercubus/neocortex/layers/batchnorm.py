from . import xp, BaseLayer
from typing import Callable

class BatchNorm(BaseLayer):
    def __init__(self, features, priority:int, sum_order, name:str, scale_shift=False ,epsilon=1e-5, momentum=0.1, dump_grad=False):
        super().__init__(name=name, trainable=scale_shift, has_bias=scale_shift, buffered=True, dump_grad=dump_grad)
        self.features, self.priority, self.sum_order, self.epsilon, self.momentum, self.scale_shift = features, priority, sum_order, epsilon, momentum, scale_shift

        self.mu = xp.zeros((self.features), dtype=xp.float32)
        self.var = xp.zeros_like(self.mu)
        self.params[f"{self.name}_mu"] = self.mu
        self.params[f"{self.name}_var"] = self.var

        if self.scale_shift:
            self.B = xp.zeros_like(self.mu)
            self.W = xp.ones(shape=self.mu.shape, dtype=xp.float32)
            self.params[f"{self.name}_W"] = self.W
            self.params[f"{self.name}_B"] = self.B
        else: self.W, self.B = 1, 0
        self.tags.extend(self.params.keys())

    # def shape(self, X):
    #     match(self.priority):
    #         case -1:
    #             return xp.moveaxis(X, (-2,-1), (0,1))
    #         case 0:
    #             return X.T
    #         case 1:
    #             return xp.moveaxis(X, (1,2), (0,1))
    #         case 2:
    #             return xp.moveaxis(X, 1, 0)
    #         case 3:
    #             return X

    #     raise Exception("Invalid priority in BatchNorm layer")

    def forward(self, X):
        # self.reshape = X.shape
        self.m = X.shape[0]
        # X = self.shape(X)
        # self.m = X[0]
        if self.training:
            self.X = X
            self.set_mu()
            self.set_var()

        # return xp.reshape(self.W * ( (X - self.mu) / (self.var + self.epsilon)**0.5 ) + self.B, self.reshape)
        return self.W * ( (X - self.mu) / (self.var + self.epsilon)**0.5 ) + self.B
    
    def set_mu(self):
        self.mu *= (1-self.momentum)
        self.mu +=  xp.sum(self.X, axis=self.sum_order, keepdims=True) / self.m * self.momentum

    def set_var(self):
        self.var *= (1-self.momentum)
        self.var += xp.sum((self.X - self.mu)**2, axis=self.sum_order, keepdims=True) / self.m * self.momentum

    def backpropagate(self, dY, optimiser=Callable):
        if self.dump_grad:
            del dY
            return 0

        # dY = self.shape(dY)
        # dz = xp.reshape(self.dL_dZ(dY), self.reshape)
        dz = self.dL_dZ(dY)
        if self.scale_shift:
            optimiser(self, self.dw(dY), self.db(dY))
        return dz 

    def db(self, dY):
        assert self.has_bias, f"Attempted backpropagation on <<{self.name}>>:{self.db.__qualname__} when scale shift not enabled"
        return xp.sum(dY, axis=self.sum_order, keepdims=True)

    def dw(self, dY):
        return xp.sum( (self.X - self.mu)\
                        / (self.var + self.epsilon)**(0.5)\
                        * dY,
                        axis=self.sum_order, keepdims=True
                    )
    
    def dL_dZ(self, dY):
        return (self.W / (self.var + self.epsilon)**0.5 / self.m)\
                * (
                    self.m * dY - xp.sum(dY, axis=self.sum_order, keepdims=True)\
                    - (self.X - self.mu) / (self.var + self.epsilon) * xp.sum(dY * (self.X - self.mu), axis=self.sum_order, keepdims=True)
                )