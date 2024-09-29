import ctypes
from ctypes import *

def getCacheOp():
    dll = ctypes.CDLL('./cacheMatMulPy.so', mode=ctypes.RTLD_GLOBAL)
    return dll.cacheTileMM

__cacheMatMul = getCacheOp()

if __name__ == '__main__':
    __cacheMatMul()
