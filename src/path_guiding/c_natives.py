from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np
from multiprocessing import RawArray

# global variable

import invoke
import pathlib

def get_shared_memory(shape, dtype=np.uint32):
	d = ctypes.c_uint32 if dtype == np.uint32 else ctypes.c_float
	shared_array_base = RawArray(d, int(np.prod(shape)))
	shared_array = np.ctypeslib.as_array(shared_array_base)
	shared_array = shared_array.reshape(shape)
	shared_array.fill(0)
	return shared_array


invoke.run(
	"gcc -shared -o libquadtree_updater.so -fPIC -fopenmp quadtree_updater.c"
)
# Load the shared library into ctypes
libname = pathlib.Path().absolute() / "libquadtree_updater.so"
c_lib = ctypes.CDLL(libname)
_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')
update_quadtree_native = c_lib.update_quadtree
update_quadtree_native.restype = None
update_quadtree_native.argtypes = [
	_doublepp,
	_doublepp,
	_doublepp,
	_doublepp,
	_doublepp,
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ctypes.c_uint32,
	ctypes.c_float
]

update_quadtree_multi_native = c_lib.update_quadtree_multi
update_quadtree_multi_native.restype = None
update_quadtree_multi_native.argtypes = [
	_doublepp,
	_doublepp,
	_doublepp,
	_doublepp,
	_doublepp,
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ctypes.c_uint32,
	ctypes.c_float
]

update_binary_tree_native = c_lib.update_binary_tree_native
update_binary_tree_native.restype = ctypes.c_uint32
update_binary_tree_native.argtypes = [
	_doublepp,
	_doublepp,
	_doublepp,
	_doublepp,
	_doublepp,
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),

	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),

	ctypes.c_uint32,
	ctypes.c_uint32,
	ctypes.c_uint32
]

update_binary_tree_native_grid = c_lib.update_binary_tree_native_grid
update_binary_tree_native_grid.restype = ctypes.c_uint32
update_binary_tree_native_grid.argtypes = [
	_doublepp,
	ctypes.c_uint32,

	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),
	ndpointer(dtype=ctypes.c_uint32, flags='C_CONTIGUOUS'),

	ctypes.c_uint32,
	ctypes.c_uint32,
	ctypes.c_uint32
]

def pp(x):
	xpp = (x.__array_interface__['data'][0] + np.arange(x.shape[0]) * x.strides[0]).astype(np.uintp)
	return xpp