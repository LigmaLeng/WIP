from . import xp, BaseLayer

class Transmogrify(BaseLayer):
    """
    Flatten layer that maintains input and output dimensions
    used for flattening and unflattening between fc and conv layers.
    """
    def __init__(self, input_dims:tuple, output_dims:int):
        super().__init__(trainable=False)
        self.input_dims, self.output_dims = input_dims, output_dims
    
    def forward(self, phi):
        return xp.reshape(phi, (phi.shape[0], self.output_dims))
    
    
    def backpropagate(self, dv, alpha, optimiser=None):
        return xp.reshape(dv, (dv.shape[0], *self.input_dims))