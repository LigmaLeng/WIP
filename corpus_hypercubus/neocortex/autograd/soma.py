# from . import xp
from dataclasses import dataclass, field, InitVar
from typing import NamedTuple, Callable, TypeVar, Any, TypeAlias, ParamSpec
from types import NotImplementedType
from collections.abc import Set
import cupy as xp
parent_module = xp.__dict__["__name__"]


S = TypeVar("S", bound="Ganglion")
P = ParamSpec("P")
Pluripotent = float | list | xp.ndarray

ALLOSTERICS = {}

class Presynaptic(NamedTuple):
    ganglion: "Ganglion"
    grad_fn: Callable[[xp.ndarray], xp.ndarray]


def differentiate(cell:Pluripotent) -> xp.ndarray:
    if isinstance(cell, xp.ndarray):
        return cell
    else:
        return xp.asarray(cell, dtype=xp.float32)




@dataclass
class Ganglion:
    data: InitVar[Pluripotent]
    is_plastic: bool = False
    synapses: Set[Presynaptic] = None
    grad: S = None
    _op: str = ""

    def __post_init__(self, data:Pluripotent) -> None:
        self.data = differentiate(data)
        # Possibly move to primitve wrapping
        self.shape = self.data.shape

    def __repr__(self) -> str:
        return f"Ganglion({self.data}, grad={self.grad}, is_plastic={self.is_plastic})"
    
    def __array_function__(self, fn, types, args, kwargs) -> Callable[..., S] | NotImplementedType:
        if fn not in ALLOSTERICS:
            return NotImplemented
        # Fallback handling for when __array_function__
        # is not overriden by Ganglion class or other numpy subclasses or containers
        if not all(issubclass(_type, Ganglion) for _type in types):
            return NotImplemented
        return ALLOSTERICS[fn](*args, **kwargs)

def modulates(enzyme):
    """Registers __array_function__ defintions for Ganglion class."""
    def conform(effector):
        ALLOSTERICS[enzyme] = effector
        return effector
    return conform

@modulates(xp.mean)
def mean(arr, *args, **kwargs):
    mu = xp.mean(arr.data, *args, **kwargs)
    return Ganglion(mu, _op="mu")
    
#     # def sum(self) -> S:
#     #     raise NotImplementedError