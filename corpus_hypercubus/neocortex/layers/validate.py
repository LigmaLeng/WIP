from typing import Callable
PAD_KW = ["valid", "same", "auto"]

def chk_pad(padding: int|tuple|str, d1=False) -> tuple:
    try:
        if _is_kwpad(padding):
           return (padding, )*2
        if isinstance(padding, int):
            return ((padding, )*2, )*2
        if isinstance(padding, tuple):
            if isinstance(padding[0], tuple):
                if isinstance(padding[1], tuple):
                    return (_chk_tuple(padding[0]), _chk_tuple(padding[1]))
                raise TypeError
            else:
                if d1:
                    return (_chk_tuple(padding), )*2
                else:
                    h_pad, w_pad = _chk_tuple(padding)
                    return (h_pad,)*2 , (w_pad,)*2
        else:
            raise TypeError
    except TypeError:
        raise TypeError(f"Invalid argument type:{type(padding)} for padding. Accepted types:\n - keywords[{'|'.join(PAD_KW)}]\n - int\n - tuple[int,int]\n - tuple[ tuple[int,int], tuple[int,int] ]")

def _is_kwpad(term: str)->bool:
    if term in PAD_KW:
        return True
    if isinstance(term, str):
        raise ValueError(f"Invalid padding keyword, accepted keywords include: [{'|'.join(PAD_KW)}]")
    return False

def _may_dupe(fn) -> Callable[[int|tuple,bool], tuple[int,int]|int]:
    def check(term:int|tuple, dupe:bool=True) -> tuple[int,int]|int:
        if isinstance(term, int):
            return (term, )*2 if dupe else term
        elif isinstance(term, tuple):
            return _chk_tuple(term) if dupe else _chk_tuple(term)[0]
        else:
            fn(term, dupe)
    return check


@_may_dupe
def chk_strides(strides:int|tuple, dupe:bool=True) -> tuple[int,int]|int:
    raise TypeError(f"Invalid argument type {type(strides)} for strides. Accepted types [int|tuple]")

@_may_dupe
def chk_kernel(dims:int|tuple, dupe:bool=True) -> tuple[int,int]|int:
    raise TypeError(f"Invalid argument type: {type(dims)} for kernel dimensions. Accepted types [int|tuple]")

def _chk_tuple(term:tuple) -> tuple[int, int]:
    if len(term) == 2 and isinstance(term[0], int) and isinstance(term[1], int):
        return (int(term[0]), int(term[1]))
    else:
        raise TypeError