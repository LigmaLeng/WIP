import numpy as np

# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(linewidth=np.inf)

"""
sample CNN parameters
"""

def main():
    l = Layer(n_kernels = 1, kernel_size = 3, input_size = 7, input_depth= 1)
    l.visualise_tensor("front")
    # test_kernel = np.arange(2*3**3, dtype=np.float32).reshape(2,3,3,3)
    # print(l.bpad)
    # print(np.rot90(test_kernel, 2, (-2,-1)))
    # print(np.einsum("hwjfg, jifg -> ihw", 
    #                 l.b_tensor,
    #                 np.rot90(test_kernel, 2, (-2,-1))))
    # print("kernels:\n", l.ker, "\n")
    # print("biases:\n" , l.b, "\n")
    # einsum = np.einsum("jdef, kldef-> jkl", l.ker, l.tensor)
    # print ("einsum:\n", einsum, "\n")
    # print("after bias:", einsum + l.b, "\n")



class Layer():
    def __init__(self, n_kernels, kernel_size, input_size, input_depth, bias_value=0.1, fill_offset=1):
        self.int_max = input_depth * input_size**2
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.input_depth= input_depth
        self.output_size = (input_size - kernel_size + 1)
        self.input = np.arange(self.int_max, dtype=np.float32).reshape(self.input_depth,self.input_size,self.input_size)

        # pad = np.pad(arr, 3, "wrap")
        # arr = np.ones((4,5,5), dtype=np.float32)
        # arr *= np.array([1,2,3,4], dtype=np.float32).reshape(4,1,1)

        self.ker = np.ones((self.n_kernels,
                        self.input_depth,
                        self.kernel_size,self.kernel_size),
                        dtype=np.float32)

        self.b = np.full((self.n_kernels, self.output_size, self.output_size), fill_value=bias_value, dtype=np.float32)

        for i in range(self.n_kernels):
            fill = i + fill_offset
            self.ker[i] *= np.full((self.input_depth,1,1), fill_value=fill, dtype=np.float32)
            self.b[i] *= fill

        self.bpad = np.pad((self.b), pad_width=((0,0),(2,2),(2,2)), mode="constant")
        
        self.tensor = np.lib.stride_tricks.as_strided(self.input, shape=(
                                                                self.b.shape[-2],
                                                                self.b.shape[-1],
                                                                self.ker.shape[-3],
                                                                self.ker.shape[-2],
                                                                self.ker.shape[-1]
                                                            ), 
                                                            strides=(
                                                                    self.input.strides[-2], 
                                                                    self.input.strides[-1],
                                                                    self.input.strides[0],
                                                                    self.input.strides[-2],
                                                                    self.input.strides[-1]
                                                                ))
        
        self.backtensor = np.lib.stride_tricks.as_strided(self.input, shape=(
                                                                            self.ker.shape[-2],
                                                                            self.ker.shape[-1],
                                                                            self.ker.shape[-3],
                                                                            self.b.shape[-2],
                                                                            self.b.shape[-1]
                                                                        ), 
                                                                        strides=(
                                                                                self.input.strides[-2], 
                                                                                self.input.strides[-1],
                                                                                self.input.strides[0],
                                                                                self.input.strides[-2],
                                                                                self.input.strides[-1]
                                                                            ))
        
        self.b_tensor = np.lib.stride_tricks.as_strided(self.bpad, shape=(
                                                                        self.input_size,
                                                                        self.input_size,
                                                                        self.n_kernels,
                                                                        self.ker.shape[-2],
                                                                        self.ker.shape[-1]
                                                                    ), 
                                                                    strides=(
                                                                            self.bpad.strides[-2], 
                                                                            self.bpad.strides[-1],
                                                                            self.bpad.strides[-3],
                                                                            self.bpad.strides[-2],
                                                                            self.bpad.strides[-1]
                                                                        ))
        
        
    
    def print_input(self):
        print(self.input)

    def print_strides(self):
        print(self.input.strides)

    def visualise_tensor(self, tensor="front"):
        if tensor == "front" or tensor == "bpad":
            view_size = self.kernel_size  
            if tensor == "bpad":
                vis = self.b_tensor
                out_size = self.input_size
            else:
                vis = self.tensor
                out_size = self.input_size - view_size + 1
        elif tensor == "back":
            vis = self.backtensor
            view_size = self.output_size
            out_size = self.input_size - view_size + 1
        else: return


        """
        print settings
        """
        
        char_pad = len(str(self.int_max))
        arr_strlen = ((char_pad + 4) * view_size + 1)
        box_height = (view_size + 1) * self.input_depth + 1
        arr_pad = -1 * (-(box_height * 2 - arr_strlen) // 2)
        box_width = arr_pad * 2 + arr_strlen
        border = "-" * (box_width * out_size + out_size + 1)
        line_break = ("|" + " "*box_width) * out_size + "|\n"


        d5 = ""
        for d4 in vis:
            d3stack = ""
            for i in range(d4.shape[1]):
                d2stack = ""
                for j in range(d4.shape[2]):
                    d1stack = ""
                    for k in range(d4.shape[0]):
                        d1stack += "|" + " "*arr_pad + np.array2string(d4[k][i][j], formatter={"float_kind": lambda x: np.format_float_positional(x, pad_left=char_pad, pad_right=char_pad)}) + " "*arr_pad
                    d2stack += d1stack + "|\n"
                d3stack += line_break + d2stack
            d5 += "{}\n{}{}".format(border, d3stack, line_break)
        print("5D-tensor:\n" + d5 + border)
        return


if __name__ == "__main__":
    main()
