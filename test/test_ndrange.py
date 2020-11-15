import taichi as ti
import taichi_glsl as ts


@ti.kernel
def fill(a: ti.template()):
    for I in ti.grouped(a):
        a[I] = I


@ti.kernel
def test_ndrange(a: ti.template()):
    shape = ts.vec3(a.shape)
    # print(shape.yz)
    for I in ti.grouped(ti.ndrange((64, 65), shape.y, shape.z)):
        print(a[I])


@ti.kernel
def test_ndrange2():
    l_b = ts.vec3(2, 2, 2)
    r_u = ts.vec3(4, 4, 4)
    r = [(l_b[i], r_u[i]) for i in range(len(l_b))]
    print(r)
    for I in ti.grouped(ti.ndrange(*r)):
        print(I)


if __name__ == '__main__':
    ti.init(ti.gpu, debug=True)
    # a = ti.Vector.field(3, ti.float32, (128, 128, 128))
    # print(a.shape)
    # fill(a)

    l_b = ts.vec3(2, 2, 2)
    u_r = ts.vec3(4, 4, 4)

    test_ndrange2()

    # test_ndrange(a)
