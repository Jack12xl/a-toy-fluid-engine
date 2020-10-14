import taichi as ti
from geometry import *

ti.init(ti.cpu, debug=True)

if __name__ == '__main__':
    collider_list = []
    collider_list.append(1)

    collider_idx_field = ti.field( dtype=ti.int32, shape=[2] )
    collider_idx_field.fill(0)

    @ti.kernel
    def test():
        print(collider_list[collider_idx_field[1]])

    test()