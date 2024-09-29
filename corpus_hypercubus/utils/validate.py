from . import xp
import numbers

def check_dice(seed):
    if not seed:
        return xp.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return xp.random.RandomState(seed)
    if isinstance(seed, xp.random.RandomState):
        return seed
    raise ValueError("Invalid rng seed")