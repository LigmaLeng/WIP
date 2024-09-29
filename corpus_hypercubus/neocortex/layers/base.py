from . import activations
from typing import Callable


class BaseLayer:
    """
    Base class for hidden layers in network.
    
    Edit: For comments in subsequent layer subclasses, I wil attempt to keep
    the naming of variables and mathematical derivations consistent over the evolutions of the code over time.
    Derivations typically refer to:
    - Input feature tensors as X/in
    - Weight and kernel tensors as W
    - Bias tensors as B
    - Output value vectors as Y/out
    """
    def __init__(self, activation=None, trainable=True, name=None, has_bias=True, buffered=False, dump_grad=False, multiheaded=False):
        self.multiheaded = multiheaded
        self.training=True
        self.trainable=trainable
        self.buffered=buffered
        if trainable or buffered:
            if not buffered:
                if not activation:
                    activation_class = getattr(activations, "Linear")
                elif activation not in activations.listed:
                    raise ValueError("Invalid activation argument. Referer to available activations as listed below:{0}".format('\n -'.join(activations.listed)))
                activation_class = getattr(activations, activation)
                self.activation = activation_class()
            self.name = name
            self.dump_grad=dump_grad
            self.has_bias=has_bias
            self.buffered=buffered
            self.params={}
            self.tags=[]

    def forward(self, X):
        raise NotImplemented

    def backpropagate(self, dY, optimiser=Callable):
        raise NotImplemented

    def link(self, insight=None, mode="transfer"):
        if not self.trainable and not self.buffered:
            raise Exception("No trainable parameters to link")
        elif mode == "transfer":
            return self.params
        elif mode == "receive":
            for tag in self.tags:
                if tag[-1]=="W":
                    self.W = insight[tag]
                    self.params[tag]=self.W
                elif tag[-1]=="B":
                    self.B = insight[tag]
                    self.params[tag]=self.B
                elif tag[-1]=="u":
                    self.mu = insight[tag]
                    self.params[tag]=self.mu
                elif tag[-1]=="r":
                    self.var = insight[tag]
                    self.params[tag]=self.var
    
    def toggle_training(self, training:bool):
        self.training=training