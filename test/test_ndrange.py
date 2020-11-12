import taichi as ti
import taichi_glsl as ts
@ti.kernel
def fill(a : ti.template()):
    for I in ti.grouped(a):
        a[I] = I

@ti.kernel
def test_ndrange(a: ti.template()):
    shape = ts.vec3( a.shape )
    # print(shape.yz)
    for I in ti.grouped(ti.ndrange((64, 65), shape.y, shape.z)):
        print(a[I])


if __name__ == '__main__':
    ti.init(ti.gpu, debug=True)
    a = ti.Vector.field(3, ti.float32, (128, 128, 128))
    # print(a.shape)
    fill(a)

    test_ndrange(a)
