import taichi as ti
import taichi_glsl as ts
from Grid import DataGrid

ti.init(ti.gpu)


@ti.kernel
def fill_data(input: ti.template()):
    for I in ti.grouped(input):
        input[I] = I


@ti.kernel
def test_kern(grid_input: ti.template()):
    for I in ti.grouped(grid_input):
        print(grid_input[I])


if __name__ == '__main__':
    a_field = ti.Vector.field(2, dtype=ti.f32, shape=[3, 3])
    a = ti.Vector([2, 3])

    grid_a = DataGrid(a_field)
    fill_data(a_field)
    test_kern(grid_a)
