from . import xp
import numpy as np
from typing import Tuple

def one_hot(input:object|list, classes:int):
    one_hot_label = xp.zeros((len(input), classes), dtype=bool)
    for i, label in enumerate(input):
        one_hot_label[i, int(label)] = 1
    return one_hot_label

def bin_hot(input:object|list, classes:int):
    rank = classes - 1
    bin_hot_label = xp.zeros((len(input), rank), dtype=bool)
    for i in range(rank):
        args = xp.argwhere(input > i)
        bin_hot_label[args, i] = 1
    assert (bin_hot_label.sum(axis=1) == input).all, "binhot error"
    return bin_hot_label

def numpyPack(*args:xp.ndarray)->Tuple[np.ndarray,...]:
    ret = []
    for array in args:
        ret.append(xp.asnumpy(array))
    return (*ret,)