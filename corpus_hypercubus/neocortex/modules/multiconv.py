from . import xp, BaseLayer, Conv2D, Conv1D, MaxPool1D
from typing import Callable

class Conv_MultiKernel_MaxPool(BaseLayer):
    def __init__(self, input_dims: tuple,
                 kernel_dims: list[tuple],
                 kernel_depth: int,
                 pad:list[tuple]|str,
                 pool_dim:int,
                 pool_stride:int,
                 activation:str,
                 has_bias=True,
                 dump_grad=False,
                 w_init="kaiming_normal",
                 name="default"):
        super().__init__(name=name, has_bias=has_bias, dump_grad=dump_grad, activation=activation, multiheaded=True)
        self.in_c = kernel_depth
        self.in_dims = input_dims
        self.pool_stride = pool_stride
        self.sublayers = [Conv2D(input_dims, k_dims, kernel_depth, pad=pad[i], activation=activation, has_bias=has_bias, dump_grad=dump_grad, w_init=w_init, name=self.name+f"{i+1}") for i, k_dims in enumerate(kernel_dims)]
        self.stack={conv:MaxPool1D(input_dims=(conv.out_c, conv.out_h), kernel_size=pool_dim, strides=pool_stride) for i, conv in enumerate(self.sublayers)}
        for conv in self.sublayers:
            self.params.update(conv.link())
            self.tags.extend(conv.tags)

    def forward(self, X):
        shp = X.shape[0]
        if X.ndim == 4:
            X = (X,)*len(self.sublayers)
        else:
            X = xp.moveaxis(X, 1, 0)

        Y = xp.empty((len(self.sublayers), shp, self.in_c, int(self.in_dims[-2]/self.pool_stride), 1))
        for i, (conv, pool) in enumerate(self.stack.items()):
            Y[i]= pool.forward(conv.forward(X[i]))

        return xp.moveaxis(Y, 1, 0)

    def backpropagate(self, dY, optimiser=Callable):
        dX = xp.empty(((len(self.sublayers), dY.shape[0], ) + self.in_dims))
        dY = xp.moveaxis(dY, 1, 0)
        for i, (conv, pool) in enumerate(self.stack.items()):
            out = pool.backpropagate(dY[i], optimiser)
            if self.dump_grad:
                conv.backpropagate(out, optimiser)
            else:
                dX[i] = conv.backpropagate(out, optimiser)
        if self.dump_grad:
            del dY
            return 0
        return xp.moveaxis(dX, 1, 0)

    def link(self, insight=None, mode="transfer"):
        if not self.trainable and not self.buffered:
            raise Exception("No trainable parameters to link")
        elif mode == "transfer":
            return self.params
        
        elif mode == "receive":
            for conv in self.sublayers:
                conv.link({tag:insight[tag] for tag in conv.tags}, mode="receive")
                self.params.update(conv.link())
    
    def toggle_training(self, training: bool):
        for conv in self.sublayers:
            conv.toggle_training(training)

