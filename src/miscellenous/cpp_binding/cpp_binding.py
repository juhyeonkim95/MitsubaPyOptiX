# ctypes_test.py
import ctypes
import pathlib
import invoke
import pybind11
print(pybind11.__file__)
print(pybind11.get_include())


def run(cpp_name, extension_name):
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "
        "`python3 -m pybind11 --includes` "
        "-I ~/anaconda3/envs/pathtracer3.8/include -I .  "
        "{0} "
        "-o {1}`python3.8-config --extension-suffix` ".format(cpp_name, extension_name)
        # "-L. -lcppmult -Wl,-rpath,.".format(cpp_name, extension_name)
    )


run("pybind11_wrapper.cpp", "libcppmult")


#
# if __name__ == "__main__":
#     # Load the shared library into ctypes
#     libname = pathlib.Path().absolute() / "libcmult.so"
#     c_lib = ctypes.CDLL(libname)
#     c_lib.cmult.restype = ctypes.c_float
#     x, y = 6, 2.3
#     answer = c_lib.cmult(x, ctypes.c_float(y))
#     #print(answer)
#     print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
