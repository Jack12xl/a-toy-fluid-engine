import taichi as ti
import taichi_glsl as ts
from Grid import CellGrid, FaceGrid
from utils import bufferPair

ti.init(ti.gpu, debug=True)


@ti.kernel
def fill_data(input: ti.template()):
    for I in ti.static(input):
        input[I] = I


@ti.kernel
def test_kern(grid_input: ti.template()):
    for I in ti.static(grid_input):
        print(grid_input[I])


@ti.kernel
def test_iter(foo: ti.template()):
    # for I in ti.static(foo):
    #     print(I)

    for I in ti.static(foo):
        print(foo.interpolate(foo.getW(I)))


if __name__ == '__main__':
    a_field = ti.Vector.field(2, dtype=ti.f32, shape=[3, 3])
    # a = ti.Vector([2, 3])
    #
    grid_a = CellGrid(a_field, dim=2, dx=ts.vec2(1.0), o=ts.vec2(0.0))
    fill_data(grid_a)
    test_kern(grid_a)
    # F = FaceGrid(ti.f32, [4, 4, 4], 3, ts.vec3(2.0), ts.vec3(0.0))
    # G = FaceGrid(ti.f32, [4, 4, 4], 3, ts.vec3(2.0), ts.vec3(0.0))
    # F.fill(ts.vec3(2.0))
    # G.fill(ts.vec3(4.0))
    #
    # p = bufferPair(F, G)
    # F.fields[0].fill(ts.vec(1.0))
    #
    # p.swap()
    # test_iter(p.nxt)
