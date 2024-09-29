from corpus_hypercubus import cp,\
                    Adam, Eve,\
                    Dense, Transmogrify, Conv_MultiKernel_GlobalPool, Conv_MultiKernel_MaxPool,\
                    Squared_Error_Objective, Leaky_ReLU, ReLU, Dropout, BatchNorm

from corpus_hypercubus.data import *
from corpus_hypercubus import Printing_press as pp
from dataclasses import dataclass, field
from typing import Optional


"""
Raw
len:  (7999,)
Raw_valid
len:  (1737,)
Raw_test
len:  (1738,)
NaN in rawfile idx: 7418  JOB-2019-0002789
CUPY_ACCELERATORS=cub,cutensor python cnn.py
    0           1         2          3          4          5          6          7          8         9
[0.9986267, 1.0268292, 0.991202, 0.9247399, 0.9625752, 1.2478939, 0.9226067, 1.0580688, 1.0483618, 0.899775 ]
"""
MAX_SEQ_LEN=256
EMBED_DIM=300
OUTPUT_SIZE = 1
NETWORK = [
            Conv_MultiKernel_MaxPool(input_dims=(1, MAX_SEQ_LEN, EMBED_DIM),
                                        kernel_dims=[(1,EMBED_DIM), (2,EMBED_DIM), (3,EMBED_DIM), (5,EMBED_DIM)],
                                        kernel_depth=64,
                                        pool_dims=[(4,1), (3,1), (2,1), (3,1)],
                                        dump_grad=True,
                                        name="conv1_"),
            Leaky_ReLU(nested=True),
            Conv_MultiKernel_GlobalPool(input_dims=[(64,64,1), (64,85,1), (64,127,1), (64,84,1)],
                                        kernel_dims=[(3,1), (6,1), (9,1), (6,1)],
                                        kernel_depth=128,
                                        name="conv2_"),
            Leaky_ReLU(),
            Transmogrify((512,1,1), 512),
            BatchNorm((512,1), 0, 1, scale_shift=False, momentum=0.6, name="bn1"),
            # Dropout(p=.5),
            Dense(512, 384, name="fc1"),
            ReLU(),
            Dense(384, OUTPUT_SIZE, name="fc2"),
        ]

def run():
    model = Model(network=NETWORK, optimiser=Adam(max_norm=3),
                  loss_func=Squared_Error_Objective.loss, loss_grad=Squared_Error_Objective.grad,
                  minibatch=512, suffix="cnn")
    model.fit(Custom.train, Custom.valid, Raw.train_labels, Raw.valid_labels, fn=cp.load)
    # s = ["seed1", "seed2", "seed3", "seed4", "seed5", "seedbsf"]
    # for i in range(len(s)):
    #     s[i] = "parameters/cnn/"+s[i]
    # model.model.weight_average(s)
    # model.upload("merged")
    # model.download("seed118")
    # model.descend_the_gradient(epochs=10000, min_lr=1e-5, max_lr=3e-4, cycle_size=50, decay=0.99, save_threshold=0.12)
    model.lr_range_search(epochs=6, min_lr=1e-5, max_lr=1e-3)


@dataclass
class Model(Eve):
    def descend_the_gradient(self, epochs:int, min_lr, max_lr, decay=1, cycle_size=1, growth_factor=1, load=None, save_threshold=0.12):
        if load!=None:
            self.download(load)
        ledger = Log.Performance_ledger()
        scheduler = Log.Cosine_annealing(self.train_instances, self.minibatch, cycle_size, min_lr, max_lr, decay, growth_factor)
        self.optimiser.scheduler = scheduler
        valid_predictions, valid_labels, valid_loss, valid_error = self.test()
        ledger.record(valid_losses=valid_loss, valid_errors=valid_error,
                      valid_predictions=valid_predictions, valid_labels=valid_labels)
        print(f"Initialised Error: {valid_error}")
        best=valid_error

        for e in range(1, epochs+1):
            train_loss, train_error = self.train(callback=scheduler.step)
            valid_predictions, valid_labels, valid_loss, valid_error = self.test()

            if scheduler.check_cycle(current_epoch=e):
                if valid_error < best and load!=None:
                    self.optimiser.zero_moments()
                    self.download(load)
                elif self.optimiser.amsgrad:
                    self.optimiser.zero_ams()

            if valid_error > best or valid_error > save_threshold:
                save_path = f"e{e}_{int(valid_error*1000):d}"
                self.upload(save_path)
                if valid_error > best:
                    best=valid_error
                    if load: load=save_path


            pp.validation((e), valid_loss, valid_error)
            ledger.record(train_losses=train_loss, train_errors=train_error,
                          valid_losses=valid_loss, valid_errors=valid_error,
                          valid_predictions=valid_predictions, valid_labels=valid_labels)
            ledger.save(f"evals/learning_rate/{self.suffix}_valid")
            scheduler.save(f"evals/learning_rate/{self.suffix}_iteration")

    
    def lr_range_search(self, epochs:int, min_lr:float, max_lr:float):
        scheduler = Log.Range_finder(self.train_instances, self.minibatch, epochs, min_lr, max_lr)
        self.optimiser.scheduler = scheduler
        for e in range(epochs):
                self.train(callback=scheduler.step)
        scheduler.save(f"evals/learning_rate/{self.suffix}_range")

    def validate_saved_model(self, filename:str):
        self.download(filename=filename)
        predictions, labels, loss, error = self.validate()
        pp.validation((0), loss, error)
        # cp.savez(file=f"evals/report/{self.suffix}", predictions=predictions,labels=labels, loss=loss, error=error)
    
    def download(self, filename:str="params"):
        self._download(file_path=f"parameters/cnn/{filename}")

    def upload(self, filename:str="params"):
        self._upload(file_path=f"parameters/cnn/{filename}")


class Log:
    class Performance_ledger():
        def __init__(self):
                self.whitelist = ["train_losses","valid_losses","valid_errors","train_errors",\
                                  "train_predictions","train_labels","valid_predictions","valid_labels",\
                                    "learning_rate"]
                self._ledger = {}
                for name in self.whitelist:
                    self._ledger[name] = []
        
        def record(self, **kwargs):
            for key, value in kwargs.items():
                if key not in self.whitelist:
                    raise Exception("_ledger page not in whitelist")
                else: self._ledger[key].append(value)
            
        def save(self, file_path:str):
            for key in self.whitelist:
                if key not in self._ledger.keys():
                    self._ledger.pop(key)

            cp.savez(file=file_path, **self._ledger)

    @dataclass
    class Learning_rate_base:
        train_instances: int
        minibatch: int
        cycle_size: int
        min_lr: float
        max_lr: float
        _lr: float = field(init=False)
        ledger: object = field(init=False)

        def __post_init__(self):
            self.updates_per_epoch = self.train_instances//self.minibatch + 1 if self.train_instances % self.minibatch != 0 else self.train_instances//self.minibatch
            self.ledger = Log.Performance_ledger()
            self._restart(self.cycle_size)

        def save(self, arg):
            self.ledger.save(arg)

        def _restart(self, cycle_size:int):
            self.steps = 0
            self.cycle_size = cycle_size

        @property
        def lr(self):
            return self._lr
        
        def _step(self, **kwargs):
            self.ledger.record(**kwargs)
            self.steps+=1

        @property
        def _run_progress(self):
            return self.steps / (self.updates_per_epoch * self.cycle_size)


    @dataclass
    class Cosine_annealing(Learning_rate_base):
        decay: float|int = field(default=1)
        growth_factor: Optional[int|float] = field(default=1)

        def __post_init__(self):
            super().__post_init__()
            self._lr = self.max_lr
            self.threshold = self.cycle_size

        def step(self, loss):
            self._step(train_losses = loss, learning_rate = self._lr)
            self._lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr)  * float(1 + cp.cos(self._run_progress * cp.pi))

        def check_cycle(self, current_epoch):
            if current_epoch == self.threshold:
                self._restart(cycle_size = int(cp.ceil(self.cycle_size * self.growth_factor)))
                # self.min_lr*=self.decay
                self.max_lr*=self.decay
                self.threshold += self.cycle_size
                return True
            return False
        
    @dataclass
    class Range_finder(Learning_rate_base):
        def __post_init__(self):
            super().__post_init__()
            self._lr = self.min_lr

        def step(self, loss):
            self._step(train_losses = loss, learning_rate = self._lr)
            self._lr = self._run_progress * (self.max_lr - self.min_lr) + self.min_lr

if __name__ == "__main__":
    run()

