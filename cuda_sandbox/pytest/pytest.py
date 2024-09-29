import ctypes
from ctypes import *

def callOp():
	dll = ctypes.CDLL('./hwg.so', mode=ctypes.RTLD_GLOBAL)
	fn = dll.hwg
	return fn

__hwg = callOp()

if __name__ == '__main__':
	__hwg()
	exit()

