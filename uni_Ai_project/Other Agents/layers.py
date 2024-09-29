import cupy as np

class Layer:
    """
    For subsequent layer subclasses, the input feature vectors will be denoted as Phi
    and the output value vectors will be denoted as V
    """
    def __init__(self, name="default"):
        self.name = name
        self.phi = None
        self.v = None
        self.m_dw = 0
        self.v_dw = 0
        self.m_db = 0
        self.v_db = 0

    def forward(self, phi):
        raise Exception("layer forwarding not implemented")

    def backpropagate(self, dE_dV, optimiser=object):
        """
        dE_dV: gradient of the error with respect to output for a layer
        alpha: learning rate
        """
        raise Exception("layer backpropagation not implemented")
    
    def link(self, insight=None, mode="update"):
        pass



class Dense(Layer):
    """
    Fully connected layer that handels functions relating to output V 
    where V = Theta . Phi + B
    and B is a vector for the biases
    and Phi is the feature vector
    """
    def __init__(self, phi_size, v_size, name="default"):
        super().__init__(name=name)
        """
        If output V is a column vector of jx1
        and input Phi is a column vector of ix1
        The weight vector Theta should then be a matrix of size jxi
        As B measures the bias for every output, it should match the shape of V : jx1
        """
        
        self.theta = np.random.randn(v_size, phi_size, dtype=np.float32)
        self.b = np.random.randn(v_size, 1, dtype=np.float32)
        # self.theta = np.random.randn(v_size, phi_size)
        # self.b = np.random.randn(v_size, 1)

    def forward(self, phi):
        self.phi = phi
        
        return np.dot(self.theta, self.phi) + self.b
    
    def backpropagate(self, dE_dV, optimiser:object):
        """
        !! DERIVING DIFFERENTIALS !!
        [dE_dPhi]
        is the derivative of the error with respsect to the input vector Phi
        following chain rule if: 
            dE_dPhi = dE_dV * dV_dPhi

        then:
            dE_dphi_i = dE_dV * dV_dphi_i
        
        since the gradient of the output of each element v_j in V with respect to each element phi_i in Phi
        excludes all other terms that don't include the feature phi_i.
        then:
            dv_j_dphi_i = (d_dphi_i) theta_ji * phi_i
                        = theta_ji

        hence by applying the inner product:
            dE_dPhi = dE_dV . Theta

        but since the shape of dE_dPhi should be ix1
        the shape of Theta would have to be transposed from jxi to ixj
        and rearranging to get
            dE_dPhi = Theta_trans . dE_dV 
        where the shape of dE_dPhi is obtained from (ixj) . (jx1)
        """
        
        dE_dPhi = np.dot(np.transpose(self.theta), dE_dV)
        self.theta, self.b = optimiser.update(*self.link(dE_dV=dE_dV), self)
        if self.name == "first":
            return 0
        return dE_dPhi
    

    def link(self, dE_dV=0, insight=None, mode="update"):
        """
        ===========================================================================================
        [dE_dB]
        is the derivative of the error with respsect to the biases vector B.
        following the same chain rules above:
            dE_dB = dE_dV * dV_dB

        since the exponent of b_j is 1, the deriative of v_j with respect to b_j is also 1
        hence:
            dE_dB = dE_dV
        ===========================================================================================
        [dE_dTheta]
        is the derivative of the error with respsect to the paramaters Theta.
        following the same chain rules above:
            dE_dTheta = dE_dV * dV_dTheta

        then:
            dE_dTheta_ji = dE_dv_j * dv_j_dtheta_ji

        since v_j is b_j + the dot product of theta_j and phi,
        the deriative of v_j with respect to theta_ji would exclude all terms not including theta_ji, 
        therefore:
            dE_dTheta_ji = dE_dv_j * phi_i

        with matrix multiplication you can get the dot product of the 2 vectors by:
            dE_dTheta = dE_dV . Phi_trans
            where Phi_trans is the transposed vector of Phi to yield a dE_dTheta with shape jxi
        
        the same is achieved with einstein sum to simplify calculations and memory costs below:
        ===========================================================================================
        """
        # pass parameters and gradients to optimiser for update
        if mode == "update":
            return self.theta, self.b, np.einsum("ja, ia -> ji", dE_dV, self.phi), dE_dV
        elif mode == "transfer":
            return [self.theta, self.b]
        elif mode == "receive":
            self.theta, self.b = insight[0], insight[1]




class Conv2D(Layer):
    """
    Convolutional layer that handels functions relating to kernels Kappa
    where output V = Phi_corr_Kappa + B
    and _corr_ is the the cross-correlation of the input Phi by the kernel Kappa
    and the convolution operation which is the cross correlation of a kernel rotated by 180 degrees is only used during backpropagation.
    and B is a matrix for the biases untied to the input
    and Phi is the feature matrix
    """
    def __init__(self, phi_dims: tuple, kappa_dim_2d: tuple, kappa_j: int, name="default"):
        super().__init__(name=name)
        phi_i, phi_h, phi_w = phi_dims
        kappa_f, kappa_g = kappa_dim_2d
        """
        NOTES ON THE KERNELS KAPPA:
        The depth of the kernel is the number of kernels j that sample the feature matrix Phi
        Where each kernel Kappa samples from the same receptive field in all channels i of Phi and sums the input to its kernel output V_j
        Where the V_j is equal to the number of kernel layers in Kappa, i.e. the depth of the output is the same as the number of kernel layers
        """
        self.kappa_j = kappa_j
        self.phi_dims = phi_dims
        self.phi_i = phi_i
        """
        For an image with (h x w) dimensions and a kernel/filter with (f x g) dimensions
        the 2D shape of the output from a 2d cross correlation would be defined as (h - f + 1) x (w - g + 1) defined here as (k x l)
        hence the kernel dimensions (j x i x f x g) and dimensions Phi (i x h x w)
        results in output dimensions v_dims denoted by (j x k x l) where the Hadamard products are summed over i for each kl in V
        """
        self.v_dims = (self.kappa_j, (phi_h - kappa_f + 1), (phi_w - kappa_g + 1))
        self.kappa_dims = (self.kappa_j, self.phi_i, kappa_f, kappa_g)

        # Initialising weights/parameters for kernel and biases
        
        # self.kappa = np.random.randn(*self.kappa_dims, dtype=np.float32)
        # self.b = np.random.randn(*self.v_dims, dtype=np.float32)
        self.kappa = np.random.randn(*self.kappa_dims)
        self.b = np.random.randn(*self.v_dims)
        self.zeroes = kappa_f - 1


    def forward(self, phi):
        self.phi = phi
        # Copying biases to later add the results of convolution operation
        
        self.v = np.copy(self.b)
        
        """
        ON COMPUTATION OF CROSS CORRELATION OF InpUT PHI AND KERNELS KAPPA: 
        SEE self.phi_tensor_backward()
        """
        self.v += np.einsum("jifg, klifg -> jkl", self.kappa, self.phi_tensor_forward())
        return self.v
    
    def phi_tensor_forward(self):
        """
        For a 3D input Phi of shape (i x h x w), 4D kernel of shape (j x i x f x g), and 3D output of shape (j x k x l)
        the shape of the 5th dimensional tensor used to compute the cross correlation between Phi and Kappa is denoted as (k x l x i x f x g)
        where shape of the window/receptive-field is the (i x f x g), i.e. a 2D window of the same shape as the kernel for every input unit i in Phi.
        The tensor is then obtained by sliding the receptive field across the dimensions of the Phi based on the selected stride (1 in this case),
        and summing the Hadamard products for every input unit Phi_i per stride, per output unit V_j

        The 5D tensor was unit tested to confirm validity with the following visualisation for a (2 x 5 x 5) input and a (3 x 3) kernel unit:
        
                InpUT            =======>                      WINDOWS/RECEPTIVE FIELDS:
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
        return np.lib.stride_tricks.as_strided(self.phi, shape=(
                                                                self.v_dims[-2],
                                                                self.v_dims[-1],
                                                                self.phi_i,
                                                                self.kappa_dims[-2],
                                                                self.kappa_dims[-1]
                                                            ), 
                                                            strides=(
                                                                    self.phi.strides[-2], 
                                                                    self.phi.strides[-1],
                                                                    self.phi.strides[-3],
                                                                    self.phi.strides[-2],
                                                                    self.phi.strides[-1]
                                                                ))

    def backpropagate(self, dE_dV, optimiser=object):
        #COMPUTE dE_dPhi before updating weights and biases
        """
        ON COMPUTATION OF CROSS CORRELATION OF InpUT PHI WITH DE_DV && CONVOLUTION OF DE_DV WITH KERNELS KAPPA
        SEE self.phi_tensor_backward() && self.dE_dV_tensor_backward() RESPECTIVELY
        """
        
        if self.name != "first":
            dE_dPhi = np.einsum("hwjfg, jifg -> ihw", 
                                self.dE_dV_tensor_backward(np.pad((dE_dV), 
                                                            pad_width=((0,0),
                                                            (self.zeroes,self.zeroes),
                                                            (self.zeroes,self.zeroes)),
                                                            mode="constant")), 
                                np.rot90(self.kappa, 2, (-2,-1)))

        self.kappa, self.b = optimiser.update(*self.link(dE_dV=dE_dV), self)
        if self.name == "first":
            return 0
        return dE_dPhi
    
    def link(self, dE_dV=0, insight=None, mode="update"):
        
        if mode == "update":
            return self.kappa, self.b, np.einsum("jkl, fgikl -> jifg", dE_dV, self.phi_tensor_backward()), dE_dV
        elif mode == "transfer":
            return [self.kappa, self.b]
        elif mode == "receive":
            self.kappa, self.b = insight[0], insight[1]

    
    def phi_tensor_backward(self):
        """
        With calculating the gradient of the error with respect to the input Phi, it might be tempting to take the error gradient wrs. to the output
        and apply the chain rule similar to derivations from the dense layer adapted to the added dimensions.
        But caution must be taken due to the fact that the gradient with respect to the output dE_dV 
        is the sum of the gradients dV_j_dKappa_ji, but the latter term is the derivative of a matrix wrs. to an undefined tensor of higher rank.
        Therefore, we have to apply another cross correlation step derive the error with respect to a kernel unit with respect to the cross correlation in the forward function.

        The intuition for the 5D tensor used to update the weights and biases are the same as the forward tensor except
        the shape of the tensor is now (f x g x i x k x l).
        notice that the shape is the shape kind of like the inversion of the forward tensor (k x l x i x f x g) which is pretty neat.
        the resulting shape works out to match the shape of the kernels because if you take any edge of the output from the first dimension
        with the length (n - f + 1), and get the resulting length after a cross correlation,
        it would compute to:
        n - (n - f + 1) + 1 = f
        """
        return np.lib.stride_tricks.as_strided(self.phi, shape=(
                                                                self.kappa_dims[-2],
                                                                self.kappa_dims[-1],
                                                                self.phi_i,
                                                                self.v_dims[-2],
                                                                self.v_dims[-1]
                                                            ), 
                                                            strides=(
                                                                    self.phi.strides[-2], 
                                                                    self.phi.strides[-1],
                                                                    self.phi.strides[-3],
                                                                    self.phi.strides[-2],
                                                                    self.phi.strides[-1]
                                                                ))
        
    def dE_dV_tensor_backward(self, dE_dV):
        return np.lib.stride_tricks.as_strided(dE_dV, shape=(
                                                        self.phi_dims[-2],
                                                        self.phi_dims[-1],
                                                        self.kappa_j,
                                                        self.kappa_dims[-2],
                                                        self.kappa_dims[-1]
                                                    ), 
                                                    strides=(
                                                            dE_dV.strides[-2], 
                                                            dE_dV.strides[-1],
                                                            dE_dV.strides[-3],
                                                            dE_dV.strides[-2],
                                                            dE_dV.strides[-1]
                                                        ))
    
class Transmogrify(Layer):
    """
    Nothing fancy, just an adaptor between layers maintain input and output dimensions
    mainly used for flattening and unflattening between fc and conv layers.
    but the most boring layer shouldn't go unnoticed so I gave it a personable name.
    """
    def __init__(self, phi_dims, v_dims, name="default", mode="dual"):
        self.modes = {"dual", "b", "f"}
        self.mode = mode
        if mode not in self.modes:
            raise ValueError("Unexpected argument in Transmogrify(mode={})".format(mode))
        super().__init__(name=name)
        self.phi_dims, self.v_dims = phi_dims, v_dims
    
    def forward(self, phi):
        if self.mode == "b":
            return phi
        return np.reshape(phi, self.v_dims)
    
    def backpropagate(self, dE_dV, alpha, optimiser=None):
        if self.mode == "f":
            return dE_dV
        return np.reshape(dE_dV, self.phi_dims)
    

class Dueling(Layer):
    def __init__(self, phi_size, q_size, name="dueling"):
        super().__init__(name=name)
        
        self.value = Dense(phi_size, 1, name="value")
        self.advantage = Dense(phi_size, q_size, name="advantage")

    def forward(self, phi, mode="quality"):
        self.phi = phi
        if mode == "quality":
            a = self.advantage.forward(self.phi)
            return (self.value.forward(self.phi) + (a - np.mean(a)))
        else:
            return self.advantage.forward(self.phi)
    
    def backpropagate(self, dE_dV, optimiser:object):

        dE_dPhi_A = self.advantage.backpropagate(dE_dV, optimiser)
        dE_dPhi_V = self.value.backpropagate(np.reshape(np.mean(dE_dV), (1,1)), optimiser)
        return ((dE_dPhi_A + dE_dPhi_V) / np.sqrt(2))
    

    def link(self, dE_dV=0, insight=None, mode="transfer"):
        if mode == "transfer":
            value_weights, value_bias = self.value.link(mode=mode)
            advantage_weights, advantage_bias = self.advantage.link(mode=mode)
            return (value_weights, value_bias, advantage_weights, advantage_bias)
        elif mode == "receive":
            self.value.link(insight=(insight[0], insight[1]), mode=mode)
            self.advantage.link(insight=(insight[2], insight[3]), mode=mode)
            return 


