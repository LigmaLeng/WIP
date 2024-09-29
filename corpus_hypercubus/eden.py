from . import xp
from . import PrintingPress as pp, one_hot, bin_hot
from typing import Optional, Callable
from dataclasses import dataclass, field

@dataclass
class Eve():
    network: list
    optimiser: object
    objective: Callable[...,object]
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

    def fit(self, x_train:str|object, y_train:str|object, x_test:str|object, y_test:str|object, fn:Optional[Callable[...,object]]=None):
        self.x_train, self.y_train, self.x_test, self.y_test = [xp.load(sample) if isinstance(sample, str) else sample for sample in [x_train, y_train, x_test, y_test]]
        self.train_instances = int(self.y_train.shape[0])
        self.test_instances = int(self.y_test.shape[0])
        self.y_train = self.y_train.astype(xp.float32)
        self.y_test = self.y_test.astype(xp.float32)

    def weight_sample_distribution(self):
        _, label_counts = xp.unique(self.y_train, return_counts=True)
        num_classes = label_counts.size
        weights = 1 / (label_counts * num_classes)
        subsample_prob = self.y_train
        for i in range(num_classes):
            subsample_prob = xp.where(subsample_prob == i, weights[i], subsample_prob)
        self.subsample_prob = subsample_prob

    def encode(self, method:str):
        if method not in ["one_hot", "bin_hot"]:
            raise ValueError("Available encoding methods: <one_hot>, <bin_hot>")
        fn = one_hot if method == "one_hot" else bin_hot
        self.y_train = fn(self.y_train, self.k_classes)
        self.y_test = fn(self.y_test, self.k_classes)

    """
    Iterator function. Eve class is initialised with list of network layers as list
    to be able to call this function for feed forward processes. 
    """
    def forward(self, input):
        output = input
        for layer in self.network:
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
        aggregate_loss, aggregate_accuracy= 0, 0

        # shuffle = xp.random.randint(self.train_instances, size=(self.train_instances))
        # shuffle = xp.random.choice(xp.arange(self.train_instances), self.train_instances, replace=True, p=self.subsample_prob)
        shuffle = xp.random.permutation(self.train_instances)
        updates_per_epoch= self.train_instances//self.minibatch + 1 if self.train_instances % self.minibatch != 0 else self.train_instances//self.minibatch
        updates = 0
        lines = pp.training(self._current_batch, self.optimiser.lr, 0, 0)
        while updates < updates_per_epoch:
            updates+=1
            if updates <= self.train_instances//self.minibatch:
                lim = updates*self.minibatch
            else: lim = self.train_instances
            idx = shuffle[(updates - 1)*self.minibatch : lim]

            loss, accuracy = self._train_on_batch(self.x_train[idx], self.y_train[idx])

            aggregate_loss+=loss
            aggregate_accuracy+=accuracy
            pp.clear(lines)
            pp.training(self._current_batch, self.optimiser.lr, loss, updates)
            callback(loss, accuracy)
        pp.clear(lines)
        return aggregate_loss/updates, aggregate_accuracy/updates

    def _train_on_batch(self, x_train, y_train):
        self._current_batch = int(x_train.shape[0])
        _x_train = xp.asarray(x_train)
        _y_train = xp.asarray(y_train)
        if self._current_batch != _y_train.shape[0]:
            raise Exception("Train and test set lengths do not match")
        
        output = self.forward(_x_train)

        if xp.isnan(output).any():
            raise RuntimeError("Nan present in training output")

        # loss, probas = self.objective.loss(_y_train, output)
        loss = self.objective.loss(_y_train, output)
        if xp.isnan(loss).any():
            raise RuntimeError("Nan present in training loss")

        # mean_loss = loss / self._current_batch

        accuracy = (xp.argmax(_y_train, axis=1) == xp.argmax(output, axis=1)).sum() / self._current_batch

        # predictions = (probas > 0.5).sum(axis=1)
        # accuracy = ( _y_train == predictions ).sum() / self._current_batch

        self._backpropagate(self.objective.grad(_y_train, output))

        self.optimiser.step()

        return float(loss), float(accuracy)
    
    def test(self, test_batch=512):
        self._toggle_training(False)
        batches_per_test = self.test_instances//test_batch + 1 if self.test_instances % test_batch != 0 else self.test_instances//test_batch
        batch=0
        predicted_val = []
        labels = []
        loss = 0
        accuracy = 0
        while batch < batches_per_test:
            batch+=1
            if batch <= self.test_instances//test_batch:
                lim = batch*test_batch
            else: lim = self.test_instances

            output = self.forward(xp.asarray(self.x_test[(batch - 1)*test_batch : lim]))

            y = xp.asarray(self.y_test[(batch - 1) * test_batch : lim])
            
            # _loss, probas = self.objective.loss(y, output)
            _loss = self.objective.loss(y, output)
            loss += _loss
            
            # predictions = (probas > 0.5).sum(axis=1)
            predictions = xp.argmax(output, axis=1)
            y = xp.argmax(y, axis=1)

            accuracy += (predictions == y).sum() / self.test_instances

            predicted_val.extend(predictions.get().tolist())
            labels.extend(y.get().tolist())

        return predicted_val, labels, float(loss), float(accuracy)
    
    def _weight_labels(self, sample_labels):
        label_counts = xp.sum(sample_labels, axis=0)
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
                layer.link({**self.stage},mode="receive")

    def _upload(self,file_path:str):
        self._stage_params()
        xp.savez(file=file_path, **self.stage)
 
    def _download(self, file_path:str):
        self._stage_params()
        params = xp.load(f"{file_path}.npz")
        for tag in self.stage.keys():
            self.stage[tag] = params[tag]
        self._unstage_params()

    def weight_average(self,path_list:list[str]):
        self._stage_params()
        for i in range(len(path_list)):
            params = xp.load(f"{path_list[i]}.npz")
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
    #     l2 = xp.sqrt(xp.sum(w**2, axis=axis, keepdims=True))
    #     scale = xp.clip(l2, 0, threshold) / (epsilon + l2)
    #     return scale
        



