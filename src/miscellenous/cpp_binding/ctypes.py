# ctypes_test.py
import ctypes
import pathlib
import invoke
from numpy.ctypeslib import ndpointer
import numpy as np


if __name__ == "__main__":
    invoke.run(
        "gcc -shared -o libcmult.so -fPIC cmult.c"
    )
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "libcmult.so"
    c_lib = ctypes.CDLL(libname)
    c_lib.cmult.restype = ctypes.c_float
    x, y = 6, 2.3
    answer = c_lib.cmult(x, ctypes.c_float(y))
    print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")

    c_lib.value = 5

    fun = c_lib.cfun
    fun.restype = None
    fun.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    ctypes.c_size_t,
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    indata = np.ones((5, 6))
    outdata = np.empty((5, 6))
    fun(indata, indata.size, outdata)

    print(outdata)
