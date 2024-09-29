from . import xp, BaseLayer
from . import functools as F
from . import validate
from typing import Callable

class MaxPool1D(BaseLayer):
    def __init__(self, input_dims: tuple, kernel_size:int, pad:int|str="auto", strides:int|tuple=None):
        super().__init__(trainable=False)
        if len(input_dims) == 2:
            self.in_c, self.in_h = input_dims
        else:
            self.in_c, self.in_h, _ = input_dims
        self.kernel_h = validate.chk_kernel(kernel_size, dupe=False)
        if not strides:
            self.stride = self.kernel_h 
        else:
            self.stride = validate.chk_strides(strides, dupe=False)

        self.out_h, self.h_pad = F.Pad.eval(self.in_h, self.kernel_h, pad, self.stride, d1=True)
        self.pad = (self.h_pad, (0, 0))

    def forward(self, X):
        window = self.metatensor_forward(F.Pad.array(X, self.pad, value=-1))

        self.mask_args = xp.nanargmax(window, axis=-1, keepdims=1) + xp.arange(self.out_h)[:,None] * self.stride - self.h_pad[0]

        return X[xp.arange(X.shape[0])[:,None,None,None],
                xp.arange(self.in_c)[:,None,None],
                self.mask_args,
                0]

    def backpropagate(self, dY, optimiser=Callable):
        unpool = xp.zeros((dY.shape[0], self.in_c, self.in_h, 1), dtype=xp.float32)
        unpool[xp.arange(dY.shape[0])[:,None,None,None],
                xp.arange(self.in_c)[:,None,None],
                self.mask_args,
                0] = dY
        # xp.put_along_axis(unpool, self.mask_args[...,None], dY, axis=-2)
        return unpool
    
    def metatensor_forward(self, X):
        batch_stride, channel_stride, h_stride, w_stride = X.strides 
        return xp.lib.stride_tricks.as_strided(X, shape=(X.shape[0],
                                                         self.in_c,
                                                         self.out_h,
                                                         self.kernel_h), 
                                                    strides=(
                                                            batch_stride,
                                                            channel_stride,
                                                            h_stride*self.stride, 
                                                            w_stride
                                                        ))

    
class MaxPool2D(BaseLayer):
    def __init__(self, input_dims:tuple, kernel_dims:int|tuple, pad:int|str="auto", strides:int|float=None):
        super().__init__(trainable=False)
        self.in_c, self.in_h, self.in_w = input_dims
        self.kernel_h, self.kernel_w = validate.chk_kernel(kernel_dims)
        if not strides:
            self.h_stride, self.w_stride = strides = self.kernel_h, self.kernel_w
        else:
            self.h_stride, self.w_stride = validate.chk_strides(strides)
        self.out_h, self.h_pad, self.out_w, self.w_pad = F.Pad.eval((self.in_h, self.in_w), (self.kernel_h, self.kernel_w), pad, strides)
        self.pad = (self.h_pad, self.w_pad)

    def forward(self, X):
        window = self.metatensor_forward(F.Pad.array(X, self.pad, value=-1))
        idx = xp.nanargmax(window.reshape(window.shape[:-2] + (self.kernel_h* self.kernel_w, )), axis=-1)
        unraveled = xp.unravel_index(xp.moveaxis(idx, -1, -3), (self.kernel_h, self.kernel_w))
        self.mask_args = (xp.arange(X.shape[0])[:,None,None,None],
                          xp.arange(self.in_c)[:,None,None],
                          unraveled[0] + xp.arange(self.out_h)[:,None] * self.h_stride - self.h_pad[0],
                          unraveled[1] + xp.arange(self.out_w) * self.w_stride - self.w_pad[0])
        
        # self.mask_args = xp.ravel_multi_index((unraveled[0] + xp.arange(self.out_h)[:,None] * self.h_stride - self.h_pad[0],\
        #                                  unraveled[1] + xp.arange(self.out_w) * self.w_stride - self.w_pad[0]),\
        #                                     (self.in_h, self.in_w))

        # return xp.take_along_axis(X.reshape(X.shape[:-2] + (self.in_h * self.in_w,)),\
        #                           self.mask_args.reshape(X.shape[:-2] + (self.out_h * self.out_w, )),
        #                           -1).reshape(X.shape[:-2] + (self.out_h, self.out_w, ))
        return X[self.mask_args]

    def backpropagate(self, dY, optimiser=Callable):
        unpool = xp.zeros((dY.shape[0], self.in_c, self.in_h, self.in_w), dtype=xp.float32)
        # xp.put_along_axis(unpool, self.mask_args, dY.reshape(dY.shape[:-2] + (self.out_h * self.out_w, )), axis=-1)
        unpool[self.mask_args] = dY
        # return unpool.reshape(dY.shape[:-2] + (self.in_h, self.in_w, ))
        return unpool

    def metatensor_forward(self, X):
        batch_stride, channel_stride, h_stride, w_stride = X.strides 
        return xp.lib.stride_tricks.as_strided(X, shape=(X.shape[0],
                                                         self.out_h,
                                                         self.out_w,
                                                         self.in_c,
                                                         self.kernel_h,
                                                         self.kernel_w), 
                                                    strides=(
                                                            batch_stride,
                                                            h_stride*self.h_stride, 
                                                            w_stride*self.w_stride,
                                                            channel_stride,
                                                            h_stride,
                                                            w_stride
                                                        ))
