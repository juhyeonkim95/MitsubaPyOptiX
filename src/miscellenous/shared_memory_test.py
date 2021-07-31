import numpy as np
import ctypes
from multiprocessing import RawArray, Array, Pool
import multiprocessing

_shared_array_base = RawArray(ctypes.c_int, 3 * 3)
_shared_array = np.ctypeslib.as_array(_shared_array_base)
_shared_array = _shared_array.reshape((3, 3))
_shared_array.fill(0)
#_shared_array = np.zeros((3,3))
# print("gID", hex(id(_shared_array)))
#_shared_array = np.zeros((3,3))

class Temp:
    def __init__(self):
        _shared_array[0] = 2
        # global _shared_array
        # shared_array_base = RawArray(ctypes.c_int, 3*3)
        # shared_array = np.ctypeslib.as_array(shared_array_base)
        # _shared_array = shared_array.reshape((3,3))
        # _shared_array.fill(3)
        #_shared_array.fill(0)
        #self.shared_array = _shared_array
        #print("ID", hex(id(self.shared_array)))

    def do_print(self, a):
        #print("ID", hex(id(self.shared_array)))
        #print("gID", hex(id(_shared_array)))

        #self.shared_array[a // 3, a % 3] = 3
        _shared_array[a // 3, a % 3] = 3
        #print(a, self.shared_array)

    def run_multi(self):
        with Pool(4) as p:
            p.map(self.do_print, [_ for _ in range(9)])

a = Temp()
a.run_multi()
#print(a.shared_array)
print(_shared_array)