import taichi as ti
import numpy as np

ti.init(ti.cpu, debug=True)

@ti.data_oriented
class test_class:
    def __init__(self):
        pass

if __name__ == '__main__':
    int_list = [1, 2]
    int_array = np.array(int_list)

    class_list = []
    class_list.append(test_class())
    class_array = np.array(class_list)

    idx_field = ti.field(dtype=ti.int32, shape=[2])
    idx_field.fill(0)

    @ti.kernel
    def test_int_array(int_list:ti.ext_arr()):
        for I in ti.grouped(idx_field):
            print(int_list[idx_field[I]])

    @ti.kernel
    def test_class_array(class_list: ti.ext_arr()):
        for I in ti.grouped(idx_field):
            print(class_list[idx_field[I]])

    @ti.kernel
    def test_class_list():
        print(class_list[idx_field[1]])


    test_int_array(int_array) # no problem
    test_class_list() # TypeError: list indices must be integers or slices, not Expr
    test_class_array(class_array) # AssertionError: Unknown type object



