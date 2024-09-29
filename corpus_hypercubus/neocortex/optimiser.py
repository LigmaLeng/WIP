from . import xp
from dataclasses import dataclass, field, InitVar
from typing import Callable

@dataclass
class Adam():
    lr: InitVar[Callable|float] = field(default=0.001)
    beta_1: float = field(default=0.9)
    beta_2: float = field(default=0.99)
    epsilon: float = field(default=1e-8)
    max_norm: float|int = field(default=None)
    amsgrad: bool = field(default=False) # Recommended use for large output spaces where convergence is an issue
    scheduler: object = field(init=False)
    t: int = field(init=False)
    _lr: float = field(init=False)
    network: list = field(init=False)

    def __post_init__(self, lr):
        self._lr = lr
        self.t = 1
        self.scheduler = None

    @property
    def lr(self):
        return self.scheduler.lr if self.scheduler else self._lr

    def fit(self, network:list):
        for layer in network:
            if layer.trainable:
                if layer.multiheaded:
                    self.fit(layer.sublayers)
                else:
                    layer.mdw = xp.zeros_like(layer.W)
                    layer.vdw = xp.zeros_like(layer.W)
                    layer.moments = [layer.mdw, layer.vdw]
                    if self.amsgrad:
                        layer.vhat_max_w = xp.zeros_like(layer.W)
                        layer.moments.append(layer.vhat_max_w)
                    if layer.has_bias:
                        layer.mdb = xp.zeros_like(layer.B)
                        layer.vdb = xp.zeros_like(layer.B)
                        layer.moments.extend([layer.mdb, layer.vdb])
                        if self.amsgrad:
                            layer.vhat_max_b = xp.zeros_like(layer.B)
                            layer.moments.append(layer.vhat_max_b)
        self.network = network
        
    def zero_moments(self):
        self.t =1
        for layer in self.network:
            if layer.trainable:
                if layer.multiheaded:
                    for sublayer in layer.sublayers:
                        for m in sublayer.moments:
                            m *= 0
                else:
                    for m in layer.moments:
                        m *= 0
    
    def zero_ams(self):
        for layer in self.network:
            if layer.trainable:
                if layer.multiheaded:
                    for sublayer in layer.sublayers:
                        sublayer.vhat_max_w *=0
                        if sublayer.has_bias:
                            sublayer.vhat_max_b *=0
                else:
                    layer.vhat_max_w *=0
                    if layer.has_bias:
                        layer.vhat_max_b *=0
                    

    def __call__(self, layer:object, dw, db=None):
        # First and second moment gradient estimates with 
        layer.mdw = self.beta_1 * layer.mdw + (1 - self.beta_1) * dw
        layer.vdw = self.beta_2 * layer.vdw + (1 - self.beta_2) * (dw ** 2)
        if xp.isnan(layer.mdw).any():
                raise RuntimeError(f"Nan present in first grad moment of layer: {layer.name}")
        if xp.isnan(layer.vdw).any():
                raise RuntimeError(f"Nan present second grad moment of layer: {layer.name}")

        if self.amsgrad:
            # vhat_max adaptively decreases the magnitude of the weight update...
            # ...based on long term memory of the running maximum bias corrected second moment term
            layer.vhat_max_w = xp.maximum(layer.vhat_max_w, (layer.vdw / (1 - self.beta_2 ** self.t)))

            # eta is a learning rate modifier controlled by a scheduler if present...
            # ...to indirectly modify the step size without affecting the decoupled weight decay term
            layer.W -= self.lr * (layer.mdw / (1 - self.beta_1 ** self.t)) / (layer.vhat_max_w**.5 + self.epsilon) 
        else:
            layer.W -= self.lr * (layer.mdw / (1 - self.beta_1 ** self.t)) / ((layer.vdw / (1 - self.beta_2 ** self.t))**.5 + self.epsilon)

        if self.max_norm:
            axis = (3,2,1) if layer.W.ndim > 2 else 0
            layer.W *= self.get_l2_scale(layer.W, axis=axis)

        if xp.isnan(layer.W).any():
                raise RuntimeError(f"Nan present in weight vector of layer: {layer.name}")
        

        if layer.has_bias:
            layer.mdb = self.beta_1 * layer.mdb + (1 - self.beta_1) * db
            layer.vdb = self.beta_2 * layer.vdb + (1 - self.beta_2) * (db ** 2)
            if xp.isnan(layer.mdb).any():
                 raise RuntimeError(f"Nan present in first bias moment of layer: {layer.name}")
            if xp.isnan(layer.vdb).any():
                raise RuntimeError(f"Nan present in second bias moment of layer: {layer.name}")
            

            if self.amsgrad:
                layer.vhat_max_b = xp.maximum(layer.vhat_max_b, (layer.vdb / (1 - self.beta_2 ** self.t)))
                layer.B -= self.lr * (layer.mdb / (1 - self.beta_1 ** self.t)) / (layer.vhat_max_b**.5 + self.epsilon)
            else:
                layer.B -= self.lr * (layer.mdb / (1 - self.beta_1 ** self.t)) / ((layer.vdb / (1 - self.beta_2 ** self.t))**.5 + self.epsilon)

            if xp.isnan(layer.B).any():
                raise RuntimeError(f"Nan present in bias vector of layer: {layer.name}")


    def step(self):
        self.t += 1

    # if max_norm:
    #     axis = (3,2,1) if layer.W.ndim > 2 else 0
    #     layer.W *= self.get_l2_scale(layer.W, axis=axis)
    def get_l2_scale(self, w, axis, epsilon=1e-12):
        l2 = xp.sqrt(xp.sum(w**2, axis=axis, keepdims=True))
        scale = xp.clip(l2, 0, self.max_norm) / (epsilon + l2)
        return scale