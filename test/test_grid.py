import taichi as ti
import taichi_glsl as ts
from Grid import CellGrid, FaceGrid

ti.init(ti.gpu, debug=True)


@ti.kernel
def fill_data(input: ti.template()):
    for I in ti.grouped(input):
        input[I] = I


@ti.kernel
def test_kern(grid_input: ti.template()):
    for I in ti.grouped(grid_input):
        print(grid_input[I])

@ti.kernel
def test_iter(foo : ti.template()):
    # for I in ti.static(foo):
    #     print(I)

    for I in ti.grouped(foo):
        print(I)

if __name__ == '__main__':
    # a_field = ti.Vector.field(2, dtype=ti.f32, shape=[3, 3])
    # a = ti.Vector([2, 3])
    #
    # grid_a = CellGrid(a_field)
    # fill_data(a_field)
    # test_kern(grid_a)
    F = FaceGrid(ti.f32, [4, 4, 4], 3, 2.0, 1.0)
    F.fill(ts.vec3(2.0))
    # for I in F:
    #     print(I)

    # for I in ti.grouped(F):
    #     print(I)

    test_iter(F)