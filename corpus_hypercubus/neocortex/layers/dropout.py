from . import xp, BaseLayer
from typing import Callable

class Dropout(BaseLayer):
    def __init__(self, p:float=0.2):
        super().__init__(trainable=False)
        self.p = p
    
    def forward(self, X):
        if not self.training: return X
        mask = ((xp.random.rand(*X.shape) > self.p)*1).astype(xp.float32)
        mask /= (1-self.p)
        return X * mask
    
    def backpropagate(self, dY, optimiser=Callable):
        return dY
