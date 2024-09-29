from . import cp
from . import Printing_press as pp, one_hot
from typing import Optional, Callable
from dataclasses import dataclass, field

@dataclass
class Eve():
    network: list
    optimiser: object
    loss_func: Callable[...,object]
    loss_grad: Callable[...,object]
    minibatch: int
    _current_batch: int = field(init=False)
    x_train: object = field(init=False)
    x_test: object = field(init=False)
    y_train: object = field(init=False)
    y_test: object = field(init=False)
    train_instances: int = field(init=False)
    test_instances: int = field(init=False)
    stage: dict = field(init=False)
    subsample_prob: object|None = field(init=False)
    k_classes: int|None = field(default=None)
    suffix:str = field(default="nn")

    def __post_init__(self):
        self._current_batch = self.minibatch
        self.optimiser.fit(self.network)
        self.stage = {}
    
    # def postable(fn):
    #     fn.is_postable=True
    #     return fn

    def fit(self, x_train:str|object, x_test:str|object, y_train:str|object, y_test:str|object, fn:Optional[Callable[...,object]]=None):
        if not fn:
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = fn(x_train), fn(x_test), fn(y_train), fn(y_test)
        self.train_instances = int(self.y_train.shape[0])
        self.test_instances = int(self.y_test.shape[0])

    def one_hot_encode(self):
        self.y_train = one_hot(self.y_train, shape=(self.train_instances, self.k_classes))
        self.y_test = one_hot(self.y_test, shape=(self.test_instances, self.k_classes))


    def weight_sample_distribution(self):
        label_counts = cp.sum(self.y_train, axis=0)
        num_classes = label_counts.size
        weights = 1 / (label_counts * num_classes)
        subsample_prob = cp.argmax(self.y_train, axis=1)
        for i in range(num_classes):
            subsample_prob = cp.where(subsample_prob == i, weights[i], subsample_prob)
        self.subsample_prob = subsample_prob


    def get_batch_size(self):
        return self._current_batch   

    """
    Iterator function. Eve class is initialised with list of network layers as list
    to be able to call this function for feed forward processes. 
    """
    def _pred(self, network, input):
        output = input
        for layer in network:
            output = layer.forward(output)
        return output
    
    def _backpropagate(self, gradient):
        dE_dV = gradient
        for layer in reversed(self.network):
            dE_dV = layer.backpropagate(dE_dV, self.optimiser)
    """
    Training loop 
    """
    def train(self, callback:Optional[Callable[..., None]]=lambda*args:None):
        self._toggle_training(True)
        # sum_mean_loss, sum_mean_accuracy= 0, 0
        mean_loss = 0
        # shuffle = cp.random.randint(self.train_instances, size=(self.train_instances))
        # shuffle = cp.random.choice(cp.arange(self.train_instances), self.train_instances, replace=True, p=self.subsample_prob)
        shuffle = cp.random.permutation(self.train_instances)
        updates_per_epoch= self.train_instances//self.minibatch + 1 if self.train_instances % self.minibatch != 0 else self.train_instances//self.minibatch
        updates = 0
        lines = pp.training(self._current_batch, self.optimiser.lr, 0, 0)
        while updates < updates_per_epoch:
            updates+=1
            if updates <= self.train_instances//self.minibatch:
                lim = updates*self.minibatch
            else: lim = self.train_instances
            idx = shuffle[(updates - 1)*self.minibatch : lim]

            loss = self._train_on_batch(self.x_train[idx], self.y_train[idx])

            mean_loss += loss
            # sum_mean_accuracy+=accuracy

            pp.clear_line(lines)
            pp.training(self._current_batch, self.optimiser.lr, loss, updates)
            callback(loss)
        pp.clear_line(lines)
        return float(mean_loss / updates_per_epoch)
        # return sum_mean_loss/updates, sum_mean_accuracy/updates

    def _train_on_batch(self, x_train, y_train):
        self._current_batch = int(x_train.shape[0])
        _x_train = cp.asarray(x_train)
        _y_train = cp.asarray(y_train)
        if self._current_batch != _y_train.shape[0]:
            raise Exception("Train and test set lengths do not match")
        
        output = self._pred(self.network, _x_train)

        if cp.isnan(output).any():
            raise RuntimeError("Nan present in training output")

        # accuracy = cp.sum((cp.argmax(_y_train, axis=1, keepdims=True) == cp.argmax(output, axis=1, keepdims=True)) * 1) / self._current_batch
        loss = self.loss_func(_y_train, output)
        root_mean_squared_error = cp.sqrt(loss)

        if cp.isnan(loss).any():
            raise RuntimeError("Nan present in training loss")
        
        dE_dV = self.loss_grad(_y_train, output)

        # mean_loss = float(loss / self._current_batch)
        
        self._backpropagate(dE_dV)

        self.optimiser.step()

        return float(root_mean_squared_error)
    
    def test(self, test_batch=512):
        self._toggle_training(False)
        batches_per_test = self.test_instances//test_batch + 1 if self.test_instances % test_batch != 0 else self.test_instances//test_batch
        batch=0
        # predicted_val = []
        # labels = []
        mean_loss = 0
        # accuracy = 0
        while batch < batches_per_test:
            batch+=1
            if batch <= self.test_instances//test_batch:
                lim = batch*test_batch
            else: lim = self.test_instances

            output = self._pred(self.network, cp.asarray(self.x_test[(batch - 1)*test_batch : lim]))

            y = cp.asarray(self.y_test[(batch - 1)*test_batch : lim])
            
            mean_loss += cp.sqrt(self.loss_func(y, output)) 

            ## Metrics if categorical
            # accuracy += cp.sum((cp.argmax(y, axis=1, keepdims=True) == cp.argmax(output, axis=1, keepdims=True)) * 1) / self.test_instances
            # p = cp.argmax(output, axis=1).get().tolist()
            # y = cp.argmax(y, axis=1).get().tolist()
            # predicted_val.extend(p)
            # labels.extend(y)
        # return predicted_val, labels, float(loss), float(accuracy)
        return float(mean_loss / batches_per_test)

    
    def _weight_labels(self, sample_labels):
        label_counts = cp.sum(sample_labels, axis=0)
        num_classes = label_counts.size
        num_samples = sample_labels.shape[0]
        return ((num_samples) / (label_counts*num_classes).reshape(1, label_counts.size))
    
    def _stage_params(self):
        for layer in self.network:
            if layer.trainable or layer.buffered:
                self.stage.update(layer.link())

    def _unstage_params(self):
        for layer in self.network:
            if layer.trainable or layer.buffered:
                layer.link(insight={tag:self.stage[tag] for tag in layer.tags},mode="receive")

    def _upload(self,file_path:str):
        self._stage_params()
        cp.savez(file=file_path, **self.stage)
 
    def _download(self, file_path:str):
        self._stage_params()
        params = cp.load(f"{file_path}.npz")
        for tag in self.stage.keys():
            self.stage[tag] = params[tag]
        self._unstage_params()

    def weight_average(self,path_list:list[str]):
        self._stage_params()
        for i in range(len(path_list)):
            params = cp.load(f"{path_list[i]}.npz")
            for tag in self.stage.keys():
                if i == 0:
                    self.stage[tag] = params[tag]
                else:
                    self.stage[tag] += params[tag]

        for w in self.stage.values():
            w/=len(path_list)
        self._unstage_params()
    
    def _toggle_training(self, training:bool):
        for layer in self.network:
            layer.toggle_training(training)


    # if max_norm:
    #         axis = (3,2,1) if layer.w.ndim > 2 else 0
    #         layer.w *= self.get_l2_scale(layer.w, axis=axis)

    # def get_l2_scale(self, w, axis, threshold=3, epsilon=1e-8):
    #     l2 = cp.sqrt(cp.sum(w**2, axis=axis, keepdims=True))
    #     scale = cp.clip(l2, 0, threshold) / (epsilon + l2)
    #     return scale
        



