import cupy as np
from layers import Conv2D, Transmogrify, Dense, Dueling
from activations import ReLU
from losses import mse, mse_prime


class Eve():
    def __init__(self, output_size:int,alpha=0.001):
        self.network = [
                        Conv2D((4, 13, 13), (3, 3), 14, name="first"),
                        ReLU(),
                        Conv2D((14, 11, 11), (3, 3), 14, name="conv2"),
                        ReLU(),
                        Conv2D((14, 9, 9), (3, 3), 14, name="conv3"),
                        ReLU(),
                        Transmogrify((14, 7, 7), (14*7*7, 1)),
                        Dense(14*7*7, output_size, "fc1"),
                        ReLU(),
                        Dense(output_size, output_size, "fc2"),
                        ReLU(),
                        Dueling(output_size, output_size, name="dueling")
                    ]
        self.output_size = output_size
        self.loss = mse
        self.loss_prime = mse_prime
        self.adam = Adam(self.network, alpha=alpha)
        self.batch_mean_td_error=[]
        self.batch_mean_gradients=[]
        self.stage = {}


    def quality(self, input):
        output = input
        for layer in self.network:
            output = layer.forward(output)
        return output

    def advantage(self, input):
        output = input
        for layer in self.network:
            if layer.name == "dueling":
                return layer.forward(output, mode="advantage")
            output = layer.forward(output)

    def batch_predict(self, inputs, batch_size:int):
        preds = None
        toggle = False
        for i in range(batch_size):
            if toggle == False:
                preds = self.quality(inputs[i])
                toggle = True
            else:
                preds = np.concatenate((preds, self.quality(inputs[i])), axis=1)
        return preds
    
    
    def batch_train(self, x_train, y_train, batch_size:int):
        if x_train.shape[0] != y_train.shape[1]:
            raise Exception("Train and test set lengths do not match")
        batch_td_err = 0.
        dE_dV = np.zeros((batch_size, self.output_size, 1), dtype=np.float32)
        for i in range(batch_size):
            y = np.reshape(y_train[:,i], (self.output_size, 1))
            #forward
            output = self.quality(x_train[i])
            #error
            batch_td_err += self.loss(y, output)
            dE_dV[i,:] = self.loss_prime(y, output)


        #backprop
        self.batch_td_error = batch_td_err / batch_size
        self.batch_mean_td_error.append(batch_td_err)
        self.batch_mean_gradients.append(float(np.sum(dE_dV)))
        # if self.batch_norm_td_error == None:
        #     self.batch_norm_td_error, self.batch_norm_error_gradients = np.asarray([batch_td_err]), np.sum(dE_dV)/batch_size
        # else:
        #     np.append(self.batch_norm_td_error, np.asarray([batch_td_err]))
        #     np.append(self.batch_norm_error_gradients, np.sum(dE_dV)/batch_size)
        # self.upload_errors
        self.adam.i_think_you_should_try_this_fruit(dE_dV, batch_size, self.output_size)

    def batch_info(self):
        return self.batch_mean_td_error[-1], self.batch_mean_gradients[-1]

    def upload_errors(self):
        np.savez("errors_per_update", mse=np.asarray(self.batch_mean_td_error), grad=np.asarray(self.batch_mean_gradients))

    def download_erros(self):
        self.batch_norm_td_error = np.load("bn_mse")
        self.batch_norm_error_gradients = np.load("bn_mse")

    def learn_from(self, twin:object):
        for offline, online in zip(self.network, twin.network):
            offline.link(
                        insight=(online.link(mode="transfer")),
                        mode="receive"
                        )
        return
    
    def stage_params(self, layer, mode="fill"):
        if mode == "fill":
            self.stage[layer.name + "_w"], \
            self.stage[layer.name + "_b"] = layer.link(mode="transfer")

    def upload(self):
        for layer in self.network:
            if layer.name == "default":
                continue
            if layer.name == "dueling":
                self.stage_params(layer.advantage)
                self.stage_params(layer.value)
            else:
                self.stage_params(layer)
        np.savez("spellbook",\
                first_w=self.stage["first_w"], first_b=self.stage["first_b"],\
                conv2_w=self.stage["conv2_w"], conv2_b=self.stage["conv2_b"],\
                conv3_w=self.stage["conv3_w"], conv3_b=self.stage["conv3_b"],\
                fc1_w=self.stage["fc1_w"], fc1_b=self.stage["fc1_b"],\
                fc2_w=self.stage["fc2_w"], fc2_b=self.stage["fc2_b"],\
                value_w=self.stage["value_w"], value_b=self.stage["value_b"],\
                advantage_w=self.stage["advantage_w"], advantage_b=self.stage["advantage_b"])
        
    def download(self):
        params = np.load("spellbook.npz")
        for layer in self.network:
            if layer.name == "default":
                continue
            if layer.name == "dueling":
                layer.link(insight=(params["value_w"], params["value_b"],\
                                    params["advantage_w"], params["advantage_b"]),\
                                    mode="receive")
            else:
                layer.link(insight=(params[layer.name+"_w"],\
                                    params[layer.name+"_b"]),\
                                    mode="receive")




class Adam():
    def __init__(self, network:list, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.network = network


    def i_think_you_should_try_this_fruit(self, gradient, batch_size, output_size):
        for i in range(batch_size):
            self.t += 1
            dE_dV = np.reshape(gradient[i,:], (output_size, 1))
            for layer in reversed(self.network):
                dE_dV = layer.backpropagate(dE_dV, self)

    
    def update(self, w, b, dw, db, layer:object):
        layer.m_dw = self.beta_1 * layer.m_dw + (1 - self.beta_1) * dw
        layer.v_dw = self.beta_2 * layer.v_dw + (1 - self.beta_2) * (dw ** 2)

        layer.m_db = self.beta_1 * layer.m_db + (1 - self.beta_1) * db
        layer.v_db = self.beta_2 * layer.v_db + (1 - self.beta_2) * (db ** 2)

        # m_dw_prime = layer.m_dw / (1 - self.beta_1 ** self.t)
        # v_dw_prime = layer.v_dw / (1 - self.beta_2 ** self.t)

        # m_db_prime = layer.m_db / (1 - self.beta_1 ** self.t)
        # v_db_prime = layer.v_db / (1 - self.beta_2 ** self.t)

        return (w - ((layer.m_dw / (1 - self.beta_1 ** self.t))
                    / (np.sqrt( layer.v_dw / (1 - self.beta_2 ** self.t))
                                + self.epsilon))
                                * self.alpha),\
                (b - ((layer.m_db / (1 - self.beta_1 ** self.t))
                    / (np.sqrt( layer.v_db / (1 - self.beta_2 ** self.t))
                                + self.epsilon))
                                * self.alpha)