from . import xp, BaseLayer
from . import functools as F
from . import validate
from typing import Callable
from abc import abstractmethod

class ConvND(BaseLayer):
    def __init__(self, input_dims: tuple, kernel_dims: tuple, kernel_depth: int,
                pad:str|int|tuple, strides:int|tuple|None=None, activation=None, has_bias=True,
                dump_grad=False, w_init="kaiming_normal", name="default"):
        """
        Convolutional layer that handles functions relating to kernels W

        where output Y = X_corr_W + B
        -  _corr_ is the the cross-correlation of the input X by the kernels W
        - the convolution operation which is the cross correlation of a kernel rotated by 180 degrees is only used during backpropagation.
        - B is a matrix for the biases untied to the input
        - X is the feature matrix
        ===========================================================================================================================================
        For an image with (h x w) dimensions
        and a kernel/filter with (f x g) dimensions

        the 2D shape of the output from a 2d cross correlation would be defined as (h - f + 1) x (w - g + 1) defined here as (k x l)
        hence the kernel dimensions 

            (j x c x f x g) 

        and dimensions of X 
        
            (m x c x h x w) 

        for a batch size m results in output dimensions

        Y denoted by (m x j x k x l) 
        
        where the Hadamard products are summed over c for each kl in Y.
        If padding and stride is taken into account the output size is defined as ((h+2p-f) // s + 1)) x ((w+2p-g) // s + 1)) for stride s
        ===========================================================================================================================================
        NOTES ON THE KERNELS W:

        The depth of the kernel is the number of feature maps/output channels j that sample the feature matrix Phi, 

        Where each kernel W samples from the same receptive field in all channels c of X and sums the input to its kernel output yj
        Where the j is equal to the number of kernel layers in W, 
        i.e. output-depth/channels/feature-maps/kernel-"layers" are all interchangeable
        """
        super().__init__(activation=activation, name=name, has_bias=has_bias, dump_grad=dump_grad)
        self.in_c, self.in_h, self.in_w = input_dims
        self.out_c = kernel_depth
        self.evaluate_padding(kernel_dims, pad, strides)
        self._strided = self.h_stride > 1 or self.w_stride > 1
        if self._strided:
            self.row_dilator, self.col_dilator = F.Dilation.matrices2d(self.out_h, self.out_w, strides)

        fan_in = self.in_c * self.kernel_h * self.kernel_w
        fan_out = self.out_c * self.kernel_h * self.kernel_w

        self.W = F.weight_init(w_init, fan_in, fan_out, self.activation.gain, size=(self.out_c, self.in_c, self.kernel_h, self.kernel_w))
        self.params[f"{self.name}_W"] = self.W

        if self.has_bias:
            self.B = xp.zeros((1, self.out_c, self.out_h, self.out_w), dtype=xp.float32)
            self.params[f"{self.name}_B"] = self.B
        else: self.B = 0

        self.tags.extend(self.params.keys())

    @abstractmethod
    def evaluate_padding(self, kernel_dims: tuple, pad:str|int|tuple="valid", strides:int|tuple|None=None)->None:
        raise NotImplementedError

    def forward(self, X):
        """
        see Conv2d.metatensor_forward()
        for computation of forward cross correlation of input X and kernels W
        """
        X = F.Pad.array(X, self.pad)
        self.X = X if self.training else None
        return self.activation(xp.tensordot(self.W, self.metatensor_forward(X), axes=[[3,2,1], [5,4,3]]).transpose((1,0,2,3)) + self.B)
        # equivalent xp.einsum("jcfg, mklcfg -> mjkl")

    def backpropagate(self, dY, optimiser=Callable):
        """
        see Conv2d.metatensor_grad() && Conv2d.metatensor_backward()
        for computation of cross correlation of input X with dL_dY
        and convolution of dL_dY with kernels W 
        """
        dY = self.activation.grad(dY)
        dB = self.dB(dY)

        if self._strided:
                dY = self.row_dilator @ dY @ self.col_dilator
        if not self.dump_grad:
            dX = xp.tensordot(xp.rot90(self.W, 2, (-2,-1)),
                                self.metatensor_backward(F.Pad.array(dY, self.transpad)),
                                axes=[[3,2,0], [5,4,3]]).transpose((1,0,2,3))
            # equivalent xp.einsum("jcfg, mhwjfg -> mchw")

        optimiser(self, self.dW(dY), dB)
        if not self.dump_grad:
            return dX 
        else:
            del dY
            return 0
        
    def dW(self, dY):
        return xp.tensordot(dY, self.metatensor_grad(), axes=[[3,2,0],[5,4,0]])
        # equivalent xp.einsum("mjkl, mcfgkl -> jcfg")

    def dB(self, dY):
        if not self.has_bias: return None
        return xp.sum(dY, axis=0, keepdims=True)
    
    def metatensor_forward(self, X):
        """
        Function to compute a 5th Dimensional view of the 3-Dimensional input tensor X of shape (c x h x w)
        according to the 4D kernels W of shape (j x c x f x g),
        to map the inputs and kernels to the 3D output Y of shape (j x k x l).

        the shape of the Tensor used to compute the cross correlation between X and W is given by (k x l x c x f x g)
        where shape of the window/receptive-field is (c x f x g), i.e. a 2D window of the same shape as the kernel for every input channel c in X.
        The tensor is then obtained by sliding the receptive field across the dimensions of X based on the selected stride (1 in this case),
        and summing the Hadamard products for every input unit xc per stride, per output unit yj
        The 5D tensor was unit tested to confirm validity with the following visualisation for a (2 x 5 x 5) input and a (3 x 3) kernel shape:      
                INPUT           =======>                      WINDOWS/RECEPTIVE FIELDS:
        [ 0.  1.  2.  3.  4.]               -------------------------------------------------------------
        [ 5.  6.  7.  8.  9.]               |                   |                   |                   |
        [10. 11. 12. 13. 14.]               |   [ 0.  1.  2.]   |   [ 1.  2.  3.]   |   [ 2.  3.  4.]   |
        [15. 16. 17. 18. 19.]               |   [ 5.  6.  7.]   |   [ 6.  7.  8.]   |   [ 7.  8.  9.]   |
        [20. 21. 22. 23. 24.]               |   [10. 11. 12.]   |   [11. 12. 13.]   |   [12. 13. 14.]   |
                                            |                   |                   |                   |
        [25. 26. 27. 28. 29.]               |   [25. 26. 27.]   |   [26. 27. 28.]   |   [27. 28. 29.]   |
        [30. 31. 32. 33. 34.]               |   [30. 31. 32.]   |   [31. 32. 33.]   |   [32. 33. 34.]   |
        [35. 36. 37. 38. 39.]               |   [35. 36. 37.]   |   [36. 37. 38.]   |   [37. 38. 39.]   |
        [40. 41. 42. 43. 44.]               |                   |                   |                   |
        [45. 46. 47. 48. 49.]               -------------------------------------------------------------
                                            |                   |                   |                   |
                                            |   [ 5.  6.  7.]   |   [ 6.  7.  8.]   |   [ 7.  8.  9.]   |
                                            |   [10. 11. 12.]   |   [11. 12. 13.]   |   [12. 13. 14.]   |
                                            |   [15. 16. 17.]   |   [16. 17. 18.]   |   [17. 18. 19.]   |
                                            |                   |                   |                   |
                                            |   [30. 31. 32.]   |   [31. 32. 33.]   |   [32. 33. 34.]   |
                                            |   [35. 36. 37.]   |   [36. 37. 38.]   |   [37. 38. 39.]   |
                                            |   [40. 41. 42.]   |   [41. 42. 43.]   |   [42. 43. 44.]   |
                                            |                   |                   |                   |
                                            -------------------------------------------------------------
                                            |                   |                   |                   |
                                            |   [10. 11. 12.]   |   [11. 12. 13.]   |   [12. 13. 14.]   |
                                            |   [15. 16. 17.]   |   [16. 17. 18.]   |   [17. 18. 19.]   |
                                            |   [20. 21. 22.]   |   [21. 22. 23.]   |   [22. 23. 24.]   |
                                            |                   |                   |                   |
                                            |   [35. 36. 37.]   |   [36. 37. 38.]   |   [37. 38. 39.]   |
                                            |   [40. 41. 42.]   |   [41. 42. 43.]   |   [42. 43. 44.]   |
                                            |   [45. 46. 47.]   |   [46. 47. 48.]   |   [47. 48. 49.]   |
                                            |                   |                   |                   |
                                            -------------------------------------------------------------
        """
        batch_stride, channel_stride, h_stride, w_stride = X.strides 
        return xp.lib.stride_tricks.as_strided(X, shape=(
                                                                X.shape[0],
                                                                self.out_h,
                                                                self.out_w,
                                                                self.in_c,
                                                                self.kernel_h,
                                                                self.kernel_w,                                             
                                                            ), 
                                                            strides=(
                                                                    batch_stride,
                                                                    h_stride*self.h_stride, 
                                                                    w_stride*self.w_stride,
                                                                    channel_stride,
                                                                    h_stride,
                                                                    w_stride
                                                                ))

    
    def metatensor_grad(self):
        """
        Function to compute strided view of input tensor to map input and incoming gradient tenseor for wight updates

        With calculating the gradient of the error with respect to the input X, it might be tempting to take the error gradient wrs. to the output
        and apply the chain rule similar to derivations from the dense layer adapted to the added dimensions.
        But caution must be taken due to the fact that the gradient with respect to the output dL_dY
        is the sum of the gradients dyj_dwjc, but the latter term is the derivative of a matrix wrs. to an undefined tensor of higher rank.
        i.e. a kernel with shape (f x g) is responsible for multiple outputs from sliding across the input matrix,
        Therefore, the derivative of a function with respect to a single unit in a kernel e.g. f1_g1 are the coefficients of the kernel during cross correlation.

        The intuition for the 5D tensor used to update the weights and biases are the same as the forward tensor except
        the shape of the tensor is now (c x f x g x k x l).
        notice that the shape is the shape kind of like the inversion of the forward tensor (k x l x c x f x g) which is pretty neat.
        This coincidence is due to the fact that the input units involved in cross correlation with a kernel unit give a shape identical to the output shape due to the sliding window
        Therefore, to find the input coefficients we just have to apply another cross correlation to the input for each channel c but with the shape (k x l) of the output.
        """
        batch_stride, channel_stride, h_stride, w_stride= self.X.strides 
        if self._strided:
            view_dims = (self.out_h + (self.out_h - 1) * self.h_stride,  self.out_w + (self.out_w - 1) * self.w_stride)
        else:
            view_dims = (self.out_h, self.out_w)
        return xp.lib.stride_tricks.as_strided(self.X, shape=(self.X.shape[0],self.in_c,self.kernel_h,self.kernel_w, *view_dims), 
                                                        strides=(
                                                                batch_stride,
                                                                channel_stride,
                                                                h_stride,
                                                                w_stride,
                                                                h_stride,
                                                                w_stride,
                                                            ))     
    
    def metatensor_backward(self, dY):
        """
        To recover the shape of the derivitive wrs. to the input after convolution for weight updates, dL_dY has to be padded to fit a view of the gradients from the kernels responsible
        for a specific node output, Hence the dY passed to this function is expected to be padded.

        Interestingly to recover the the derivative, an actual convolution has to be applied where the kernel is rotated by 180 degrees as the edges of the inputs
        contribute to fewer outputs and to recover the derivative wrs. to the input the kernel is rotated to match the contribution of a kernel unit with the output unit.
        """
        batch_stride, channel_stride, h_stride, w_stride= dY.strides 
        return xp.lib.stride_tricks.as_strided(dY, shape=(
                                                        dY.shape[0],
                                                        self.in_h,
                                                        self.in_w,
                                                        self.out_c,
                                                        self.kernel_h,
                                                        self.kernel_w,
                                                    ), 
                                                    strides=(
                                                            batch_stride,
                                                            h_stride, 
                                                            w_stride,
                                                            channel_stride,
                                                            h_stride,
                                                            w_stride,
                                                        ))

class Conv2D(ConvND):
    def __init__(self, input_dims: tuple, kernel_dims: tuple, kernel_depth: int, pad:str|int|tuple="valid", strides:int|tuple|None=None, activation=None, has_bias=True, dump_grad=False, w_init="kaiming_normal", name="default"):
        super().__init__(input_dims, kernel_dims, kernel_depth, pad, strides, activation, has_bias, dump_grad, w_init, name)

    def evaluate_padding(self, kernel_dims: tuple, pad: str|int|tuple, strides:int|tuple=None)->None:
        self.kernel_h, self.kernel_w = validate.chk_kernel(kernel_dims)
        if not strides:
            self.h_stride, self.w_stride = strides = (1,1)
        else:
            self.h_stride, self.w_stride = validate.chk_strides(strides)

        self.out_h, self.h_pad, self.out_w, self.w_pad = F.Pad.eval((self.in_h, self.in_w), (self.kernel_h, self.kernel_w), pad, strides)
        self.pad = (self.h_pad, self.w_pad)

        self.transpad = ((self.kernel_h - 1 - self.h_pad[0], self.kernel_h - 1 - self.h_pad[1]),
                            (self.kernel_w - 1 - self.w_pad[0], self.kernel_w - 1 - self.w_pad[1]))

class Conv1D(ConvND):
    def __init__(self, input_dims: tuple, kernel_size: int, kernel_depth: int, pad:str|int|tuple="valid", strides:int|None=None, activation=None, has_bias=True, dump_grad=False, w_init="kaiming_normal", name="default"):
        if len(input_dims) == 2:
            input_dims = input_dims + (1,)
        super().__init__(input_dims, kernel_size, kernel_depth, pad, strides, activation, has_bias, dump_grad, w_init, name)

    def evaluate_padding(self, kernel_dims:int, pad:str|int|tuple, strides:int|None=None)->None:
        self.kernel_h = validate.chk_kernel(kernel_dims, dupe=False)
        if not strides:
            self.h_stride = 1
        else:
            self.h_stride = validate.chk_strides(strides, dupe=False)
        self.w_stride = self.kernel_w = self.out_w = 1

        self.out_h, self.h_pad = F.Pad.eval(self.in_h, self.kernel_h, pad, self.h_stride, d1=True)
        self.pad = (self.h_pad, (0, 0))
        self.transpad = ((self.kernel_h - 1 - self.h_pad[0], self.kernel_h - 1 - self.h_pad[1]),
                            (0, 0))