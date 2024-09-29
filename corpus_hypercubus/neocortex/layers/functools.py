from . import xp
import math
from .validate import chk_pad, PAD_KW

__all__=["Pad", "Dilation", "weight_init"]

class Pad:
    @staticmethod
    def eval(in_dims:int|tuple,
                 kernel_dims:int|tuple,
                 padding:int|tuple|str,
                 strides:tuple|int=(1,1),
                 d1=False) -> tuple[tuple[int,tuple],...]:

        padding = chk_pad(padding, d1)
        if isinstance(in_dims, int):
            if not isinstance(kernel_dims, int) or not isinstance(strides, int):
                raise RuntimeError("Multidimensional kernels and strides not appropriate for 1D input")
            return Pad._strat(in_dims, kernel_dims, strides, padding[0])

        return Pad._strat(in_dims[0], kernel_dims[0], strides[0], padding[0]) + Pad._strat(in_dims[1], kernel_dims[1], strides[1], padding[1])

    @staticmethod
    def _strat(in_dim:int, w_dim:int, stride:int, pad:str|tuple)-> tuple[int, tuple]:
        if pad in PAD_KW:
            if pad == "valid":
                return Pad._eval_valid_1d(in_dim, w_dim, stride)
            else:
                return Pad._eval_from_src(in_dim, w_dim, stride)

        if isinstance(pad, tuple):
            return Pad._eval_from_pad(in_dim, w_dim, stride, pad)

    @staticmethod    
    def _eval_valid_1d(in_dim:int, w_dim:int, stride:int) -> tuple[int, tuple]:
        return math.ceil((in_dim - w_dim + 1) / stride), (0, 0)

    @staticmethod
    def _eval_from_src(in_dim:int, w_dim:int, stride:int) -> tuple[int, tuple]:
        out_dim = math.ceil(in_dim / stride)
        dim_add = max((out_dim - 1) * stride + w_dim - in_dim, 0)
        return out_dim, (dim_add // 2, dim_add - (dim_add//2))

    @staticmethod
    def _eval_from_pad(in_dim:int, w_dim:int, stride:int, padding:tuple) -> tuple[int, tuple]:
        return math.ceil((in_dim + sum(padding) - w_dim)/stride + 1), padding

    @staticmethod
    def array(arr:xp.ndarray, padding:tuple[tuple[int,int], tuple[int,int]], value=0) -> xp.ndarray:
        if sum(padding[0]) + sum(padding[1]) == 0:
            return arr
        values = (xp.nan,)*2 if value == -1 else (value,)*2
        return xp.pad(arr,
                      pad_width=((0,0), (0,0), padding[0], padding[1]),
                      mode="constant", constant_values=values)

class Dilation:
    @staticmethod
    def matrices2d(out_h:int, out_w:int, strides:tuple) -> tuple[xp.ndarray, xp.ndarray]:
        assert isinstance(strides, tuple), "Invalid stride input while evalulating dilation matrix, only tuples accepted"
        return Dilation.matrix1d(out_h, strides[0]), Dilation.matrix1d(out_w, strides[1], transposed=True)

    @staticmethod
    def matrix1d(out_dim:int, stride:int, transposed:bool=False) -> xp.ndarray:
        if not (stride > 1 and out_dim > 1):
            return xp.identity(out_dim, dtype=bool)
        dilations = (out_dim-1) * stride
        matrix = xp.zeros((dilations + out_dim, out_dim), dtype=bool)
        idx = xp.arange(out_dim) + xp.arange(out_dim) * out_dim * stride
        matrix[xp.unravel_index(idx, (dilations + out_dim, out_dim))] = True
        if transposed:
            return matrix.T
        else:
            return matrix


def weight_init(mode, fan_in, fan_out, gain, size) -> xp.ndarray:
    match mode:
        case "normal" :
            lo = 0
            hi = (1 / fan_in)**.5
        case "glorot_normal" :
            lo = 0
            hi = gain * (2 / (fan_in+fan_out))**.5
        case "kaiming_normal" :
            lo = 0
            hi = gain / fan_out**.5
        case "uniform" :
            hi = (3/fan_in)**.5
        case "kaiming_uniform":
            hi = gain * (3/fan_out)**.5
        case "glorot_uniform" :
            hi = gain * (6 / (fan_in+fan_out))**.5
        case _:
            raise ValueError("Invalid argument for weight initialization")
    
    if "normal" in mode:
        return xp.random.normal(lo, hi, size=(size)).astype(xp.float32)
    else:
        return xp.random.uniform(-hi, hi, size=(size)).astype(xp.float32)