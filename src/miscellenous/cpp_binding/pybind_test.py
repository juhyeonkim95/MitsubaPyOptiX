import pybind11_example

if __name__ == "__main__":
    # Sample data for your call
    x, y = 6, 2.3

    answer = pybind11_example.cpp_function(x, y)
    print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")